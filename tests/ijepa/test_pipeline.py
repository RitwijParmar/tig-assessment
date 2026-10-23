import copy
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from ijepa import train as engine
from ijepa.utils import digest
from ijepa.model import _verify_vendor
from ijepa.data import ResearchImages
from ijepa.classifier import WeldClassifier
from ijepa.data import normalize_rows, validate_roles
from ijepa.classifier import fit_probe
from ijepa.data import TransformSpec, content_mask
from ijepa.model import IJEPAAdapter
from ijepa.vendor.ijepa.vision_transformer import (
    VisionTransformer,
    VisionTransformerPredictor,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_native_sources_are_pinned_and_corruption_fails(tmp_path, monkeypatch):
    assert (
        _verify_vendor()["source_commit"] == "52c1ae95d05f743e000e8f10a1f3a79b10cff048"
    )
    for name, expected in engine.PINNED.items():
        assert (
            hashlib.sha256((engine.UPSTREAM / name).read_bytes()).hexdigest()
            == expected
        )
        assert engine.upstream(name)
    (tmp_path / "transforms.py").write_text("changed")
    monkeypatch.setattr(engine, "UPSTREAM", tmp_path)
    with pytest.raises(ValueError, match="changed"):
        engine.upstream("transforms.py")


def test_native_masks_match_official_sampler_and_preserve_rng():
    ids = ["frame" + str(i) for i in range(8)]
    sampler = engine.NativeMasker(43)
    reference = engine.upstream("multiblock.py").MaskCollator(
        input_size=(224, 224),
        patch_size=14,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
    )
    for epoch in range(8):
        before = torch.get_rng_state().clone()
        actual = sampler.sample(ids, epoch=epoch)
        assert torch.equal(before, torch.get_rng_state())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(engine.view_seed(43, ids, epoch, 0))
            _, contexts, targets = reference([{"id": value} for value in ids])
        torch.testing.assert_close(
            actual["context_indices"], contexts[0], rtol=0, atol=0
        )
        for a, b in zip(actual["target_indices"], targets):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_content_mask_excludes_padding_intersections():
    mask = content_mask((800, 974)).reshape(16, 16)
    assert int(mask.sum()) == 192
    assert not mask[:, :2].any() and not mask[:, 14:].any()
    assert mask[:, 2:14].all()
    assert content_mask((224, 224)).all()


def test_local_metadata_rejects_recording_leakage_and_label_changes():
    roles = {}
    fingerprints = {"roles": {}}
    for i, role in enumerate(("fitting", "selection", "assessment")):
        rows = normalize_rows(
            [
                {
                    "path": f"g{i}/frame_00001.png",
                    "group": f"g{i}",
                    "frame": 1,
                    "label6": i % 2,
                    "target": i % 2,
                    "crc": 10,
                    "original_bytes": 20,
                }
            ]
        )
        roles[role] = rows
        fingerprints["roles"][role] = {
            "group_sha256": [hashlib.sha256(f"g{i}".encode()).hexdigest()],
            "frames": 1,
            "normalized_rows_sha256": digest(rows),
        }
    validate_roles(roles, fingerprints)
    leaked = copy.deepcopy(roles)
    leaked["assessment"] = leaked["fitting"]
    with pytest.raises(ValueError):
        validate_roles(leaked, fingerprints)
    altered = copy.deepcopy(roles)
    altered["selection"][0]["label6"] = 0
    with pytest.raises(ValueError):
        validate_roles(altered, fingerprints)


def test_image_checksum_failure_is_detected(tmp_path):
    file = tmp_path / "g/frame_00001.png"
    file.parent.mkdir()
    Image.new("L", (80, 90), 100).save(file)
    raw = file.read_bytes()
    row = {
        "path": "g/frame_00001.png",
        "group": "g",
        "frame": 1,
        "label6": 0,
        "target": 0,
        "crc": zlib.crc32(raw) & 0xFFFFFFFF,
        "original_bytes": len(raw),
    }
    data = ResearchImages([row], tmp_path, TransformSpec())
    assert data[0]["image"].shape == (3, 224, 224)
    changed = bytearray(raw)
    changed[-1] ^= 1
    file.write_bytes(changed)
    with pytest.raises(ValueError, match="CRC"):
        data[0]


def test_classifier_forward_preserves_encoder_and_matches_explicit_formula():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 4)

        def forward(self, image):
            return self.projection(image.flatten(2).transpose(1, 2))

    encoder = Encoder()
    head = nn.Linear(4, 1)
    mean, std = torch.arange(4).float(), torch.arange(1, 5).float()
    model = WeldClassifier(encoder, mean, std, head.state_dict())
    images = torch.randn(2, 3, 2, 2)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    with torch.no_grad():
        tokens = encoder(images)
        pooled = (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        expected = head((pooled - mean) / std).squeeze(-1)
    torch.testing.assert_close(model(images, valid), expected, rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_readout_fits_detached_features_and_uses_fitting_only_normalization(tmp_path):
    fitting = [
        {
            "path": f"a/frame_{i}.png",
            "group": "a" if i < 4 else "b",
            "frame": i,
            "label6": i % 2,
            "target": i % 2,
        }
        for i in range(8)
    ]
    selection = [
        {**row, "path": row["path"].replace("a/", "v/"), "group": "v" if i < 4 else "w"}
        for i, row in enumerate(fitting)
    ]
    x = torch.tensor([[-2.0, -1.0] if i % 2 == 0 else [2.0, 1.0] for i in range(8)])
    y = x * 1.3
    saved = fit_probe(
        x,
        y,
        fitting,
        selection,
        tmp_path / "probe",
        "native_epoch1",
        "mean",
        torch.device("cpu"),
    )
    torch.testing.assert_close(saved["mean"], x.mean(0), rtol=0, atol=0)
    torch.testing.assert_close(saved["std"], x.std(0, unbiased=False), rtol=0, atol=0)
    assert saved["score"]["selection_BA"] == 1.0
    assert (tmp_path / "probe/best.pt").is_file()


def test_native_adaptation_epoch_completes_with_real_images_and_weight_audit(tmp_path):
    from functools import partial

    config = json.loads(
        (Path(__file__).parents[2] / "models/ijepa-vit-h14/config.json").read_text()
    )
    config["data_root"] = str(tmp_path)
    config["runtime"]["workers"] = 0
    config["training"].update(epochs=1, batch_size=4, gradient_accumulation=2)
    roles = {"fitting": [], "selection": []}
    rng = np.random.default_rng(43)
    for role, n in (("fitting", 8), ("selection", 4)):
        for i in range(n):
            path = f"{role}{i % 2}/frame_{i}.png"
            file = tmp_path / path
            file.parent.mkdir(exist_ok=True)
            Image.fromarray(
                rng.integers(0, 256, size=(180, 160, 3), dtype=np.uint8)
            ).save(file)
            raw = file.read_bytes()
            roles[role].append(
                {
                    "path": path,
                    "group": f"{role}{i % 2}",
                    "frame": i,
                    "target": i % 2,
                    "label6": i % 2,
                    "crc": zlib.crc32(raw) & 0xFFFFFFFF,
                    "original_bytes": len(raw),
                }
            )
    norm = partial(nn.LayerNorm, eps=1e-6)
    context = VisionTransformer(
        img_size=[224],
        patch_size=14,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=norm,
    )
    predictor = VisionTransformerPredictor(
        num_patches=256,
        embed_dim=32,
        predictor_embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=norm,
    )
    model = IJEPAAdapter(
        context,
        copy.deepcopy(context),
        predictor,
        config,
        {"source": "tiny native fixture"},
    )
    called = []

    def readout(teacher, epoch, updates):
        called.append((epoch, updates))
        assert teacher is model.target_encoder
        return {"selection_BA": 0.5}

    engine.train_ssl(
        config,
        roles,
        tmp_path / "adaptation",
        model,
        readout,
        lambda *args, **kwargs: None,
    )
    audit = json.loads((tmp_path / "adaptation/Weight_Audit.json").read_text())
    assert called == [(1, 1)]
    assert all(
        part["every_ordinary_tensor_changed"] for part in audit["roles"].values()
    )
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )


def test_standalone_pipeline_seals_before_assessment_and_replays_selected_model(
    tmp_path, monkeypatch
):
    from functools import partial
    from ijepa import classifier as inference, train as pipeline, evaluate as assessment

    config = json.loads(
        (Path(__file__).parents[2] / "models/ijepa-vit-h14/config.json").read_text()
    )
    config["data_root"] = str(tmp_path)
    config["runtime"]["workers"] = 0
    config["training"].update(epochs=1, batch_size=4, gradient_accumulation=2)
    config["manifest_sha256"] = "synthetic-fixture-manifest"
    roles = {}
    rng = np.random.default_rng(15)
    for role, n in (("fitting", 8), ("selection", 4), ("assessment", 4)):
        roles[role] = []
        for i in range(n):
            path = f"{role}{i % 2}/frame_{i}.png"
            file = tmp_path / path
            file.parent.mkdir(exist_ok=True)
            Image.fromarray(
                rng.integers(0, 256, size=(180, 160, 3), dtype=np.uint8)
            ).save(file)
            raw = file.read_bytes()
            roles[role].append(
                {
                    "path": path,
                    "group": f"{role}{i % 2}",
                    "frame": i,
                    "target": i % 2,
                    "label6": i % 2,
                    "crc": zlib.crc32(raw) & 0xFFFFFFFF,
                    "original_bytes": len(raw),
                }
            )
    norm = partial(nn.LayerNorm, eps=1e-6)

    def encoder(**kwargs):
        return VisionTransformer(
            img_size=[224],
            patch_size=14,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
            norm_layer=norm,
        )

    def build(config, device):
        context = encoder()
        predictor = VisionTransformerPredictor(
            num_patches=256,
            embed_dim=32,
            predictor_embed_dim=16,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
            norm_layer=norm,
        )
        return IJEPAAdapter(
            context,
            copy.deepcopy(context),
            predictor,
            config,
            {"source": "tiny full-pipeline fixture"},
        ).to(device)

    monkeypatch.setattr(pipeline, "load_config", lambda path: copy.deepcopy(config))
    monkeypatch.setattr(pipeline, "load_roles", lambda value: copy.deepcopy(roles))
    monkeypatch.setattr(assessment, "load_config", lambda path: copy.deepcopy(config))
    monkeypatch.setattr(assessment, "load_roles", lambda value: copy.deepcopy(roles))
    monkeypatch.setattr(pipeline, "build_model", build)
    monkeypatch.setattr(inference, "vit_huge", encoder)
    accessed = []
    original = ResearchImages.__getitem__

    def record_access(self, index):
        accessed.append(self.rows[index]["group"])
        return original(self, index)

    monkeypatch.setattr(ResearchImages, "__getitem__", record_access)
    run = tmp_path / "run"
    chosen = pipeline.train("synthetic-config.json", run, "cpu")
    assert chosen["adaptation_updates"] == 1
    assert not any(group.startswith("assessment") for group in accessed)
    seal = json.loads((run / "seal.json").read_text())
    assert seal["assessment_pixels_accessed"] is False
    metrics = assessment.evaluate(run, "cpu")
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert any(group.startswith("assessment") for group in accessed)
    assert (run / "assessment_metrics.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        assessment.evaluate(run, "cpu")
