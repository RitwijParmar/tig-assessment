"""Released architectures, checked weight-format conversion and native readouts."""

import importlib
from pathlib import Path
import sys
import torch
from torch import nn
from safetensors.torch import load_file
from .shared import TransformSpec


def source_import(package, source):
    source = Path(source).resolve()
    existing = sys.modules.get(package)
    if existing is not None and source not in Path(existing.__file__).resolve().parents:
        raise RuntimeError(
            "Restart Python: a different upstream " + package + " is imported"
        )
    sys.path.insert(0, str(source))
    return importlib.import_module(package)


class RGBTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image.convert("RGB"))


class Encoder(nn.Module):
    def __init__(self, model, name, readout=None):
        super().__init__()
        self.model, self.name, self.readout = model, name, readout
        self.requires_grad_(False).eval()

    def forward(self, images):
        value = self.model.forward_head(
            self.model.forward_features(images), pre_logits=True
        )
        if value.ndim != 2 or value.shape[1] != 2048 or not torch.isfinite(value).all():
            raise ValueError("Expected finite 2048-dimensional released features")
        return value.float()


def build(lock, device="cpu", geometry="native"):
    name, source = lock["model"], lock["source"]
    if name != 'mobilenetv5':
        raise ValueError("This package only builds its declared encoder")
    timm = source_import("timm", source)
    model = timm.create_model("mobilenetv5_300m.gemma3n", pretrained=False)
    from timm.models.mobilenetv5 import checkpoint_filter_fn

    model.load_state_dict(
        checkpoint_filter_fn(load_file(lock["checkpoint"]), model), strict=True
    )
    cfg = timm.data.resolve_model_data_config(model)
    if (
        cfg["input_size"] != (3, 768, 768)
        or tuple(cfg["mean"]) != (0.0, 0.0, 0.0)
        or tuple(cfg["std"]) != (1.0, 1.0, 1.0)
    ):
        raise ValueError("Released MobileNetV5 preprocessing changed")
    transform = timm.data.create_transform(**cfg, is_training=False)
    encoder = Encoder(model, name)
    common = TransformSpec(
        image_size=224, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
    )
    if geometry not in ("native", "letterbox224"):
        raise ValueError("Declare native or letterbox224 geometry")
    return encoder.to(device).eval(), (
        RGBTransform(transform) if geometry == "native" else common
    )
