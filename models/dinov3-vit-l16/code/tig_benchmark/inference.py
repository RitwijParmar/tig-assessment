"""Portable inference for the best model stored in this model folder."""

from pathlib import Path
import torch
from PIL import Image
from .assets import MODEL_DIRS, load_assets
from .models import build
from weights import read_manifest,checkpoint_file,sha256


from .shared import configure_precision
from .pipeline import head_logits

def load_best(folder, device="cpu", assets=None):
    folder = Path(folder)
    weights = folder / "weights"
    inventory=read_manifest(folder)
    descriptor = inventory["inference"]
    if descriptor["source_model_config_sha256"] != sha256(folder / "config.json"):
        raise ValueError("Checkpoint recipe changed")
    name = next(iter(MODEL_DIRS))
    lock = load_assets(name, assets or folder / "assets/source/assets.local.json", require_checkpoint=False)
    configure_precision()
    base_spec = inventory["artifacts"]["best_encoder"]
    base = checkpoint_file(weights, base_spec)
    if base_spec["sha256"] != lock["checkpoint_sha256"]:
        raise ValueError("Portable base is not the declared released encoder")
    lock = {**lock, "checkpoint": str(base.resolve())}
    encoder, transform = build(lock, device, descriptor["geometry"])
    if descriptor["encoder_format"] == "supervised_torch":
        path = checkpoint_file(weights, inventory["artifacts"]["best_encoder"])
        encoder.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)["encoder"],
            strict=True,
        )
    spec = inventory["artifacts"]["best_classifier"]
    path = checkpoint_file(weights, spec)
    probe = torch.load(path, map_location="cpu", weights_only=True)
    encoder.eval().requires_grad_(False)
    return encoder, transform, probe, descriptor


def predict_local(folder, image, device="cpu", checkpoint="best", assets=None):
    if checkpoint != "best":
        raise ValueError("Only the best checkpoint is distributed")
    encoder, transform, probe, descriptor = load_best(folder, device, assets)
    with Image.open(image) as im:
        x = transform(im.convert("RGB"))[None].to(device)
    with torch.no_grad():
        logit = float(head_logits(encoder(x).cpu(), probe)[0])
    return {
        "logit": logit,
        "defect_probability": float(torch.sigmoid(torch.tensor(logit))),
        "prediction": "defect" if logit >= 0 else "good",
        "threshold": 0.5,
        "checkpoint": checkpoint,
        "mode": descriptor["mode"],
    }


def predict(folder, image, device="cpu", assets=None):
    return predict_local(folder, image, device, "best", assets)
