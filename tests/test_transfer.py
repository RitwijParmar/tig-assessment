"""Isolation, checkpoint and upstream-readout contracts for matched transfer."""

import copy
import hashlib
import json
import zlib
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from tig_benchmark import assets, data, models, pipeline, finetune
from tig_benchmark.shared import digest


def fingerprint(roles):
    return {
        "roles": {
            k: {
                "frames": len(rows),
                "group_sha256": sorted(
                    hashlib.sha256(g.encode()).hexdigest()
                    for g in {r["group"] for r in rows}
                ),
                "normalized_rows_sha256": digest(rows),
            }
            for k, rows in roles.items()
        }
    }


@pytest.fixture
def sample(tmp_path, monkeypatch):
    roles = {role: [] for role in ("fitting", "selection", "assessment")}
    for role, rows in roles.items():
        for label in (0, 1):
            for i in range(2):
                rel = f"{role}{label}/frame_{i}.png"
                p = tmp_path / rel
                p.parent.mkdir(exist_ok=True)
                Image.fromarray(np.full((20, 24), 35 + label * 180 + i, np.uint8)).save(
                    p
                )
                raw = p.read_bytes()
                rows.append(
                    {
                        "path": rel,
                        "group": f"{role}{label}",
                        "frame": i,
                        "target": label,
                        "label6": label,
                        "crc": zlib.crc32(raw) & 0xFFFFFFFF,
                        "original_bytes": len(raw),
                    }
                )
        rows.sort(key=lambda r: r["path"])
    monkeypatch.setattr(data, "fingerprints", lambda: fingerprint(roles))
    meta = tmp_path / "registration.json"
    meta.write_text(json.dumps({"rows": sum(roles.values(), [])}))
    data.prepare(meta, tmp_path, tmp_path / "prepared")
    return tmp_path, roles


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2048)
        self.register_buffer("fixed_buffer", torch.arange(3.0))

    def forward(self, images):
        return self.layer(images.mean((2, 3)))


def tiny_build(lock, device, geometry="native"):
    torch.manual_seed(912)
    return Tiny().requires_grad_(False).eval().to(device), models.RGBTransform(
        lambda im: torch.tensor(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
    )


@pytest.fixture
def stub(sample, monkeypatch):
    root, roles = sample
    (root / "fixture.weights").write_bytes(b"fixture weights")
    monkeypatch.setattr(
        pipeline,
        "load_assets",
        lambda name, path: {"model": name, "checkpoint": str(root / "fixture.weights")},
    )
    monkeypatch.setattr(pipeline.models, "build", tiny_build)
    recipe = assets.model_config("mobilenetv5")["classifier"]
    recipe.update(epochs=5, patience=2)
    monkeypatch.setattr(
        pipeline,
        "model_config",
        lambda name: {
            "classifier": recipe,
            "extraction_batch_size": 2,
            "readout": "fixture",
        },
    )
    return root, roles


@pytest.mark.parametrize("model", ["mobilenetv5", "dinov3"])
def test_full_pipeline_separates_assessment_and_replays(stub, monkeypatch, model):
    root, roles = stub
    accessed = []
    original = pipeline.ResearchImages.__getitem__

    def access(self, index):
        accessed.append(self.rows[index]["group"])
        return original(self, index)

    monkeypatch.setattr(pipeline.ResearchImages, "__getitem__", access)
    run = root / model
    pipeline.fit(model, root / "prepared/data.local.json", "unused", run, batch_size=2)
    assert not any(g.startswith("assessment") for g in accessed)
    probe = torch.load(run / "frozen/best.pt", weights_only=True)
    cache = torch.load(run / "features.local.pt", weights_only=True)
    assert torch.equal(probe["mean"], cache["fitting"].mean(0))
    assert probe["fitting_rows_sha256"] == digest(roles["fitting"])
    assert pipeline.verify(run)["verified"]
    result = pipeline.evaluate(run)
    assert result["balanced_accuracy"] is not None
    predictions = json.loads((run / "assessment_predictions.json").read_text())
    assert [r["path"] for r in predictions] == [r["path"] for r in roles["assessment"]]
    with pytest.raises(FileExistsError):
        pipeline.evaluate(run)
    with pytest.raises(FileExistsError):
        pipeline.fit(model, root / "prepared/data.local.json", "unused", run)


@pytest.mark.parametrize(
    "artifact",
    ["config.json", "assets.local.json", "selected/classifier.pt", "replay.local.pt"],
)
def test_post_selection_tampering_is_rejected(stub, artifact):
    root, _ = stub
    run = root / "tamper"
    pipeline.fit(
        "mobilenetv5", root / "prepared/data.local.json", "unused", run, batch_size=2
    )
    with (run / artifact).open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError):
        pipeline.evaluate(run)
    assert not (run / "assessment_predictions.json").exists()


def test_changed_metadata_after_seal_is_rejected(stub):
    root, _ = stub
    run = root / "tamper"
    pipeline.fit(
        "dinov3", root / "prepared/data.local.json", "unused", run, batch_size=2
    )
    with (root / "prepared/manifest.local.json").open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError, match="metadata changed"):
        pipeline.evaluate(run)


def test_split_and_annotation_changes_are_rejected(sample):
    root, roles = sample
    config, actual = data.load_data(root / "prepared/data.local.json")
    assert actual == roles
    for field, changed in [("label6", 4), ("group", "another-recording"), ("crc", 0)]:
        bad = copy.deepcopy(actual)
        bad["fitting"][0][field] = changed
        with pytest.raises(ValueError):
            data.validate_roles(bad, fingerprint(roles))
    bad = copy.deepcopy(actual)
    bad["selection"][0] = bad["fitting"][0]
    with pytest.raises(ValueError):
        data.validate_roles(bad, fingerprint(roles))


def test_wrong_checkpoint_and_modified_source_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "weights"
    path.write_bytes(b"wrong weights")
    monkeypatch.setattr(assets, "verify_source", lambda *args: "pinned")
    with pytest.raises(ValueError, match="declared released"):
        assets.register("dinov3", tmp_path, path, tmp_path / "lock.json")
    monkeypatch.undo()
    with pytest.raises(ValueError, match="Missing pinned"):
        assets.verify_source("dinov3", tmp_path)


def test_grayscale_conversion_precedes_native_transform():
    im = Image.fromarray(np.ones((3, 4), dtype=np.uint8))
    assert models.RGBTransform(lambda value: value.mode)(im) == "RGB"


def test_dino_readout_excludes_registers_and_keeps_patch_mean(tmp_path):
    p = tmp_path / "dinov3/eval"
    p.mkdir(parents=True)
    # Fixture represents the released helper API; actual source parity is checked
    # separately on the pinned released model, not asserted by this fixture.
    (p / "linear.py").write_text(
        "def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):\n    patch, cls = x_tokens_list[-1]\n    return torch.cat((cls, patch.mean(1)), dim=1)\n"
    )

    class Tokens(nn.Module):
        def get_intermediate_layers(self, x, **kw):
            assert kw == {"n": 1, "return_class_token": True, "norm": True}
            return [(torch.ones(len(x), 196, 1024) * 3, torch.ones(len(x), 1024) * 7)]

    enc = models.Encoder(Tokens(), "dinov3", models.official_linear_input(tmp_path))
    features = enc(torch.zeros(2, 3, 224, 224))
    assert features.shape == (2, 2048)
    assert (features[:, :1024] == 7).all() and (features[:, 1024:] == 3).all()


def test_finetune_includes_epoch_zero_and_respects_warmup(stub, monkeypatch):
    root, roles = stub
    cfg = {
        "seed": 43,
        "batch_size": 2,
        "workers": 0,
        "data_root": str(root),
        "geometry": "native",
    }
    monkeypatch.setattr(finetune, "build", tiny_build)
    enc, transform = tiny_build({}, "cpu")
    features = pipeline.extract(enc, transform, roles["fitting"], cfg, "cpu")
    probe = {
        "identity": "fixture",
        "mean": features.mean(0),
        "std": features.std(0, unbiased=False).clamp_min(1e-6),
        "head": nn.Linear(2048, 1).state_dict(),
        "epoch": 0,
        "score": {"selection_BA": 1.0},
        "fitting_rows_sha256": digest(roles["fitting"]),
        "selection_rows_sha256": digest(roles["selection"]),
    }
    recipe = {
        "effective_batch": 4,
        "epochs": 3,
        "warmup_epochs": 2,
        "head_lr": 0.001,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "patience": 2,
    }
    result = finetune.run_candidate(
        {"model": "mobilenetv5"}, roles, cfg, probe, 1e-4, root / "ft", "cpu", recipe
    )
    assert result["epoch"] == 0 and result["encoder_path"] is None
    audit = json.loads((root / "ft/weight_audit.json").read_text())
    assert audit["parameters_with_gradients"] and audit["warmup_updates"] == 2
    assert (root / "ft/final_encoder.pt").exists() and not (
        root / "ft/best_encoder.pt"
    ).exists()
    assert finetune.schedule(0, 10, 3) == pytest.approx(1 / 3)
    assert finetune.schedule(3, 10, 3) == pytest.approx(1.0)


def test_existing_common_classifier_recipe_is_exact():
    for name in assets.MODEL_DIRS:
        from tig_benchmark.shared import IJEPA_ROOT

        expected = json.loads((IJEPA_ROOT / "config.json").read_text())["classifier"]
        assert assets.model_config(name)["classifier"] == expected


def test_retained_checkpoint_survives_download_cache_cleanup(tmp_path):
    from tig_benchmark.checkpoints import retain

    src = tmp_path / "downloaded"
    src.write_bytes(b"important encoder weights")
    dst = tmp_path / "selected" / "encoder.safetensors"
    retain(src, dst)
    src.unlink()
    assert dst.is_file() and not dst.is_symlink()
    assert dst.read_bytes() == b"important encoder weights"


def test_assessment_requires_completed_reload_verification(stub):
    root, _ = stub
    run = root / "not-verified"
    pipeline.fit(
        "dinov3", root / "prepared/data.local.json", "unused", run, batch_size=2
    )
    (run / "status.json").unlink()
    with pytest.raises(ValueError, match="verification must finish"):
        pipeline.evaluate(run)
    assert not (run / ".assessment.started").exists()


def test_other_installed_upstream_cannot_silently_replace_pinned_source(
    tmp_path, monkeypatch
):
    import sys, types

    module = types.ModuleType("fixture_upstream")
    module.__file__ = "/another/source/__init__.py"
    monkeypatch.setitem(sys.modules, "fixture_upstream", module)
    with pytest.raises(RuntimeError, match="different upstream"):
        models.source_import("fixture_upstream", tmp_path)
