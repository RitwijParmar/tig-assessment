"""Portable inference for the best model stored in this model folder."""

import json
from pathlib import Path
import torch
from PIL import Image
from .assets import MODEL_DIRS, load_assets
from .models import build
from .pipeline import head_logits
from .shared import configure_precision, sha256


def checkpoint_file(root, item):
    relative = Path(item["path"])
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Checkpoint path must stay inside this model's weights folder")
    if sha256(path) != item["sha256"]:
        raise ValueError("Portable checkpoint checksum mismatch")
    return path


def predict_local(folder, image, device="cpu", checkpoint="best", assets=None):
    folder = Path(folder)
    if checkpoint != "best":
        raise ValueError("Only the best checkpoint is distributed")
    weights = folder / "weights"
    descriptor = json.loads((weights / checkpoint / "model.json").read_text())
    if descriptor["format"] != "tig-checkpoint-v1" or descriptor["kind"] != checkpoint:
        raise ValueError("Invalid checkpoint descriptor")
    if descriptor["source_model_config_sha256"] != sha256(folder / "config.json"):
        raise ValueError("Checkpoint recipe changed")
    name = next(iter(MODEL_DIRS))
    lock = load_assets(name, assets or folder / "assets/pretrained/assets.local.json")
    configure_precision()
    base_spec = descriptor.get("pretrained_encoder", descriptor["encoder"])
    base = checkpoint_file(weights, base_spec)
    if base_spec["sha256"] != lock["checkpoint_sha256"]:
        raise ValueError("Portable base is not the declared released encoder")
    lock = {**lock, "checkpoint": str(base.resolve())}
    encoder, transform = build(lock, device, descriptor["geometry"])
    if descriptor["encoder_format"] == "supervised_torch":
        path = checkpoint_file(weights, descriptor["encoder"])
        encoder.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)["encoder"],
            strict=True,
        )
    spec = descriptor["classifier"]
    path = checkpoint_file(weights, spec)
    probe = torch.load(path, map_location="cpu", weights_only=True)
    with Image.open(image) as im:
        x = transform(im.convert("RGB"))[None].to(device)
    encoder.eval().requires_grad_(False)
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
