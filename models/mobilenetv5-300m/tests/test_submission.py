"""Submission keeps runnable pipelines and evaluates the saved best pair."""
import json
from pathlib import Path
import zlib
import pytest
import torch
from PIL import Image
from torch import nn
import evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_required_pipelines_and_matching_best_weights_are_present():
    manifest = json.loads((ROOT / "weights/manifest.json").read_text())
    assert set(manifest["artifacts"]) == {"best_encoder", "best_classifier"}
    assert (ROOT / "weights/README.md").is_file()
    assert not (ROOT / "weights/final").exists()
    assert (ROOT / "code/evaluation.py").is_file()
    family = manifest["inference"]["family"]
    required = ["ijepa/train.py", "ijepa/evaluate.py"] if family == "ijepa" else ["tig_benchmark/pipeline.py"]
    if family == "mobilenetv5":required.append("tig_benchmark/finetune.py")
    assert all((ROOT / "code" / p).is_file() for p in required)
    probe = torch.load(ROOT / "weights/best/classifier.pt", weights_only=True)
    assert probe["epoch"] == manifest["inference"]["classifier_epoch"]


def test_published_checkpoint_assessment_reads_only_assessment_images(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / "weights/manifest.json").read_text())
    ijepa = manifest["inference"]["family"] == "ijepa"
    roles = {}
    for role in ("fitting", "selection", "assessment"):
        rows = []
        for target in (0, 1):
            relative = f"{role}{target}/frame_0.png"
            path = tmp_path / relative
            path.parent.mkdir()
            Image.new("RGB", (224, 224), (255 * target,) * 3).save(path)
            raw = path.read_bytes()
            rows.append(dict(path=relative, group=f"{role}{target}", frame=0,
                             target=target, label6=target, crc=zlib.crc32(raw) & 0xffffffff,
                             original_bytes=len(raw)))
        roles[role] = rows
    config = {"data_root": str(tmp_path)}
    from transfer_core import data as data_module
    if not ijepa:
        from tig_benchmark import data as native_data
        monkeypatch.setattr(native_data, "load_data", lambda path: (config, roles))
    else:
        monkeypatch.setattr(data_module, "load_config", lambda path: config)
        monkeypatch.setattr(data_module, "load_roles", lambda cfg: roles)
        monkeypatch.setattr(data_module, "preprocessing", lambda cfg: data_module.TransformSpec())
    accessed = []
    images = data_module.ResearchImages
    if not ijepa:
        from tig_benchmark.shared import ResearchImages
        images = ResearchImages
    original = images.__getitem__
    def get(self, index):
        accessed.append(self.rows[index]["group"])
        return original(self, index)
    monkeypatch.setattr(images, "__getitem__", get)
    class Tiny(nn.Module):
        def forward(self, images, valid=None):
            logits = images.mean((1, 2, 3))
            return logits if ijepa else logits[:, None].expand(-1, 2048)
    probe = dict(mean=torch.zeros(2048), std=torch.ones(2048),
                 head={"weight": torch.ones(1, 2048) / 2048, "bias": torch.zeros(1)})
    loaded = Tiny() if ijepa else (Tiny(), data_module.TransformSpec(), probe, {})
    monkeypatch.setattr(evaluation, "load_best", lambda *args: loaded)
    prepared = tmp_path / "prepared.json";prepared.write_text("{}")
    output = tmp_path / "assessment-output"
    result = evaluation.evaluate_best(ROOT, prepared, output)
    assert result["balanced_accuracy"] == 1.0
    assert accessed == ["assessment0", "assessment1"]
    records = json.loads((output / "assessment_predictions.json").read_text())
    assert [row["path"] for row in records] == [row["path"] for row in roles["assessment"]]
    assert json.loads((output / "evaluation.json").read_text())["threshold"] == 0.5
    with pytest.raises(FileExistsError):evaluation.evaluate_best(ROOT, prepared, output)
    with pytest.raises(ValueError, match="positive"):
        evaluation.evaluate_best(ROOT, prepared, tmp_path / "invalid-batch", batch_size=0)


def test_training_cli_defaults_to_the_submitted_single_lr(monkeypatch):
    import sys
    from tig_benchmark import __main__ as cli
    captured = {}
    def fit(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {"checked": True}
    monkeypatch.setattr(cli.pipeline, "fit", fit)
    monkeypatch.setattr(sys, "argv", ["run.py", "train", "--model", "mobilenetv5",
                                    "--data", "prepared", "--assets", "original-pretrained",
                                    "--output", "new-run"])
    cli.main()
    assert captured["args"][8] is True
    assert captured["kwargs"]["encoder_lr"] == 1e-5


def test_source_only_asset_registration_is_for_inference_only(tmp_path, monkeypatch):
    from tig_benchmark import assets
    from tig_benchmark.shared import digest
    family = next(iter(assets.MODEL_DIRS))
    config = assets.model_config(family)
    monkeypatch.setattr(assets, "verify_source", lambda *args: "pinned")
    lock = {"format": "tig-model-assets-v1", "model": family, "source": "checked",
            "source_revision": "pinned", "checkpoint": None,
            "checkpoint_sha256": config["checkpoint_sha256"], "model_config_sha256": digest(config)}
    path = tmp_path / "assets.local.json";path.write_text(json.dumps(lock))
    assert assets.load_assets(family, path, require_checkpoint=False) == lock
    with pytest.raises(ValueError, match="Training requires"):
        assets.load_assets(family, path)
