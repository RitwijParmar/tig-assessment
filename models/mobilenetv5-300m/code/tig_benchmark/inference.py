"""Portable inference for the best/final model stored in this model folder."""

import json
from pathlib import Path
import torch
from PIL import Image
from .assets import MODEL_DIRS, load_assets
from .models import build
from .pipeline import head_logits
from .shared import configure_precision, sha256


def predict_local(folder, image, device="cpu", checkpoint="best", assets=None):
    folder = Path(folder)
    if checkpoint not in ("best", "final"):
        raise ValueError("Choose best or final")
    weights = folder / "weights" / checkpoint
    descriptor = json.loads((weights / "model.json").read_text())
    name = next(iter(MODEL_DIRS))
    lock = load_assets(name, assets or folder / "assets/pretrained/assets.local.json")
    configure_precision()
    base = weights / descriptor["pretrained_encoder"]["path"]
    if sha256(base) != descriptor["pretrained_encoder"]["sha256"]:
        raise ValueError("Portable base encoder checksum mismatch")
    lock = {**lock, "checkpoint": str(base.resolve())}
    encoder, transform = build(lock, device, descriptor["geometry"])
    if descriptor["supervised_encoder"]:
        spec = descriptor["supervised_encoder"]
        path = weights / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError("Portable tuned encoder changed")
        encoder.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)["encoder"],
            strict=True,
        )
    spec = descriptor["classifier"]
    path = weights / spec["path"]
    if sha256(path) != spec["sha256"]:
        raise ValueError("Portable classifier changed")
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
