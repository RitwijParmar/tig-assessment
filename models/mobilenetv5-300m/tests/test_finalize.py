"""Only completed, explicitly retained trials may enter stopped-search selection."""
import json
import pytest
import torch
from torch import nn

import finalize_completed as command
from tig_benchmark.shared import digest


@pytest.fixture
def completed(tmp_path, monkeypatch):
    roles = {key: [{"group": key}] for key in ("fitting", "selection", "assessment")}
    run = tmp_path / "run"
    folder = run / "supervised/lr1e-05"
    folder.mkdir(parents=True)
    (run / "frozen").mkdir()
    (run / "config.json").write_text(json.dumps({"model": "mobilenetv5", "geometry": "native"}))
    (run / "assets.local.json").write_text("{}")
    (run / "replay.local.pt").write_bytes(b"fitting fixture")
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"base encoder")
    common = {"fitting_rows_sha256": digest(roles["fitting"]), "selection_rows_sha256": digest(roles["selection"])}
    torch.save({**common, "epoch": 3, "score": {"selection_BA": .6}}, run / "frozen/best.pt")
    torch.save({**common, "epoch": 1, "score": {"selection_BA": .8}, "identity": "best"}, folder / "best_head.pt")
    torch.save({**common, "epoch": 2, "score": {"selection_BA": .75}, "identity": "final"}, folder / "final_head.pt")
    for kind in ("best", "final"):
        torch.save({"identity": kind, "encoder": nn.Linear(2, 2).state_dict()}, folder / (kind + "_encoder.pt"))
    (folder / "history.json").write_text(json.dumps([
        {"epoch": 0, "selection_BA": .6}, {"epoch": 1, "selection_BA": .8}, {"epoch": 2, "selection_BA": .75}
    ]))
    (folder / "weight_audit.json").write_text(json.dumps({"epoch0_eligible": True, "optimizer_updates": 4}))
    # An unfinished alternative must never be read or resumed.
    unfinished = run / "supervised/lr3e-05"
    unfinished.mkdir()
    (unfinished / "history.json").write_text("not a completed candidate")
    recipe = {"supervised": {"epochs": 2, "patience": 2, "encoder_lr_candidates": [1e-5, 3e-5, 1e-4]}}
    monkeypatch.setattr(command, "model_config", lambda name: recipe)
    monkeypatch.setattr(command, "load_data_payload", lambda cfg: (cfg, roles))
    monkeypatch.setattr(command, "load_assets", lambda name, path: {"checkpoint": str(base)})
    monkeypatch.setattr(command, "build", lambda *args: (nn.Linear(2, 2), None))
    monkeypatch.setattr(command, "verify", lambda *args: {"verified": True})
    return run, folder, recipe


def test_stop_selects_completed_best_and_preserves_its_final(completed):
    run, _, _ = completed
    result = command.finalize(run, [1e-5], "User stopped extra search before assessment", "cpu")
    assert result["selected"]["epoch"] == 1
    assert result["amendment"]["completed_rates_used_for_selection"] == [1e-5]
    assert result["amendment"]["skipped_rates"] == [3e-5, 1e-4]
    assert torch.load(run / "final/classifier.pt", weights_only=True)["epoch"] == 2
    assert not (run / ".assessment.started").exists()
    assert json.loads((run / "status.json").read_text())["stage"] == "sealed_and_reload_verified"
    with pytest.raises(ValueError, match="unsealed"):
        command.finalize(run, [1e-5], "repeat", "cpu")


def test_incomplete_trial_cannot_be_labeled_complete(completed):
    run, _, recipe = completed
    recipe["supervised"]["epochs"] = 3
    with pytest.raises(ValueError, match="did not finish"):
        command.finalize(run, [1e-5], "stop", "cpu")
    assert not (run / "seal.json").exists()
    assert not (run / "selected").exists()


def test_mismatched_encoder_head_is_rejected_before_sealing(completed):
    run, folder, _ = completed
    torch.save({"identity": "wrong epoch"}, folder / "best_encoder.pt")
    with pytest.raises(AssertionError):
        command.finalize(run, [1e-5], "stop", "cpu")
    assert not (run / "seal.json").exists()
