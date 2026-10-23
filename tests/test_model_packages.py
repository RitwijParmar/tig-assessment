"""Package contracts prevent mismatched model weights, data and result status."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "ijepa-vit-h14",
    "ijepa-vit-h14-pretrained",
    "mobilenetv5-300m",
    "dinov3-vit-l16",
)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_package_identity_results_and_weights_agree(model_id):
    folder = ROOT / "models" / model_id
    package = json.loads((folder / "model.json").read_text())
    config = json.loads((folder / package["configuration"]).read_text())
    result = json.loads((folder / package["results"]).read_text())
    weights = json.loads((folder / package["weights"]).read_text())
    assert package["model_id"] == weights["model_id"] == model_id
    assert (folder / package["implementation"]).is_dir()
    assert package["architecture"]["threshold"] == result["threshold"] == 0.5
    assert config["classifier"]["threshold"] == 0.5
    split = json.loads((folder / package["data"]["split_config"]).read_text())[
        "data_split"
    ]
    for role, expected in package["data"]["roles"].items():
        assert expected == {k: split["roles"][role][k] for k in expected}
    if package["status"] == "completed":
        for name in ("validation", "assessment"):
            assert 0 <= result[name]["recording_balanced_accuracy"] <= 1
        assert (
            weights["artifacts"]["classifier"]["sha256"]
            == result["weights"]["classifier_sha256"]
        )
        encoder = weights["artifacts"]["selected_encoder"]
        if package["training"]["weld_jepa_adaptation"]:
            assert encoder["sha256"] == result["weights"]["selected_encoder_sha256"]
        else:
            assert encoder["path"] is None
            assert (
                weights["artifacts"]["pretrained_encoder"]["sha256"]
                == result["weights"]["base_sha256"]
            )
    else:
        assert result["validation"] is result["assessment"] is None
        assert weights["artifacts"]["classifier"]["path"] is None
        assert (
            weights["artifacts"]["pretrained_encoder"]["sha256"]
            == config["checkpoint_sha256"]
        )


@pytest.mark.parametrize("role", ["encoder", "target_encoder"])
def test_ijepa_file_loader_selects_explicit_checkpoint_role(
    tmp_path, monkeypatch, role
):
    from ijepa import classifier
    from ijepa.utils import sha256

    original = nn.Linear(2, 2)
    selected = nn.Linear(2, 2)
    head = nn.Linear(2, 1)
    encoder_path, head_path = tmp_path / "encoder.pt", tmp_path / "classifier.pt"
    states = {"encoder": selected.state_dict(), "target_encoder": original.state_dict()}
    torch.save(states, encoder_path)
    torch.save(
        {"mean": torch.zeros(2), "std": torch.ones(2), "head": head.state_dict()},
        head_path,
    )
    monkeypatch.setattr(classifier, "vit_huge", lambda **kwargs: nn.Linear(2, 2))
    hashes = {"encoder": sha256(encoder_path), "classifier": sha256(head_path)}
    model = classifier.WeldClassifier.from_files(
        encoder_path, head_path, expected=hashes, encoder_role=role
    )
    for name, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, states[role][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="checkpoint changed"):
        classifier.WeldClassifier.from_files(
            encoder_path,
            head_path,
            expected={**hashes, "classifier": "wrong"},
            encoder_role=role,
        )
    with pytest.raises(ValueError, match="target_encoder"):
        classifier.WeldClassifier.from_files(
            encoder_path, head_path, encoder_role="context_encoder"
        )
