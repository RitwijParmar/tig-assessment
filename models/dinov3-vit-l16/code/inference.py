"""Portable inference for the best model stored in this model folder."""

import json
from pathlib import Path
import torch
from PIL import Image
from native import MODEL_DIRS,load_assets
from native import build
from torch import nn
from weights import read_manifest,checkpoint_file,sha256


def configure_precision():
    """Match the recorded float32/highest-matmul/default-TF32-convolution recipe."""
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cudnn, "conv"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        torch.backends.cudnn.allow_tf32 = True

def head_logits(features, probe):
    head = nn.Linear(features.shape[1], 1)
    head.load_state_dict(probe["head"], strict=True)
    with torch.no_grad():
        return head((features.cpu() - probe["mean"]) / probe["std"]).squeeze(-1)

def predict_local(folder, image, device="cpu", checkpoint="best", assets=None):
    folder = Path(folder)
    if checkpoint != "best":
        raise ValueError("Only the best checkpoint is distributed")
    weights = folder / "weights"
    inventory=read_manifest(folder)
    descriptor = inventory["inference"]
    if descriptor["source_model_config_sha256"] != sha256(folder / "config.json"):
        raise ValueError("Checkpoint recipe changed")
    name = next(iter(MODEL_DIRS))
    lock = load_assets(name, assets or folder / "assets/source/assets.local.json")
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


def predict(folder,image,device='cpu',assets=None):
    return predict_local(folder,image,device,'best',assets)
