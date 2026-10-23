"""Released architectures, checked weight-format conversion and native readouts."""

import ast
import importlib
import importlib.util
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


def native_transforms(source):
    file = Path(source) / "dinov3/data/transforms.py"
    spec = importlib.util.spec_from_file_location("tig_dino_native_transforms", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_linear_input(source):
    # Execute the unchanged, pure upstream readout function, without importing
    # its distributed/ImageNet training CLI and optional evaluation dependencies.
    file = Path(source) / "dinov3/eval/linear.py"
    tree = ast.parse(file.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_linear_input"
    )
    space = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), "exec"), space)
    return space["create_linear_input"]


def convert_dino_hf(state, native):
    """HF's official split-QKV safetensors -> Meta state names, no learned changes."""
    result, consumed = {}, set()

    def take(key):
        consumed.add(key)
        return state[key]

    direct = {
        "cls_token": "embeddings.cls_token",
        "storage_tokens": "embeddings.register_tokens",
        "mask_token": "embeddings.mask_token",
        "patch_embed.proj.weight": "embeddings.patch_embeddings.weight",
        "patch_embed.proj.bias": "embeddings.patch_embeddings.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    for dst, src in direct.items():
        result[dst] = take(src)
    # The official HF export adds a singleton token axis to the mask token.
    # Reshape only this known representation difference; preserve every value.
    result["mask_token"] = result["mask_token"].reshape(native["mask_token"].shape)
    for i in range(24):
        src, dst = f"layer.{i}.", f"blocks.{i}."
        result[dst + "attn.qkv.weight"] = torch.cat(
            [take(src + "attention." + q + "_proj.weight") for q in ("q", "k", "v")]
        )
        q, v = take(src + "attention.q_proj.bias"), take(src + "attention.v_proj.bias")
        result[dst + "attn.qkv.bias"] = torch.cat([q, torch.zeros_like(q), v])
        for a, b in [
            ("norm1", "norm1"),
            ("norm2", "norm2"),
            ("attn.proj", "attention.o_proj"),
            ("mlp.fc1", "mlp.up_proj"),
            ("mlp.fc2", "mlp.down_proj"),
        ]:
            for suffix in ("weight", "bias"):
                result[dst + a + "." + suffix] = take(src + b + "." + suffix)
        result[dst + "ls1.gamma"] = take(src + "layer_scale1.lambda1")
        result[dst + "ls2.gamma"] = take(src + "layer_scale2.lambda1")
        # HF omits Meta's deterministic, nonlearned key-bias masks.
        result[dst + "attn.qkv.bias_mask"] = native[dst + "attn.qkv.bias_mask"]
    result["rope_embed.periods"] = native["rope_embed.periods"]
    if consumed != set(state) or set(result) != set(native):
        raise ValueError("Unexplained DINO checkpoint keys")
    if any(result[k].shape != native[k].shape for k in native):
        raise ValueError("DINO checkpoint shape mismatch")
    return result


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
        if self.name == "mobilenetv5":
            value = self.model.forward_head(
                self.model.forward_features(images), pre_logits=True
            )
        else:
            tokens = self.model.get_intermediate_layers(
                images, n=1, return_class_token=True, norm=True
            )
            value = self.readout(tokens, use_n_blocks=1, use_avgpool=True)
        if value.ndim != 2 or value.shape[1] != 2048 or not torch.isfinite(value).all():
            raise ValueError("Expected finite 2048-dimensional released features")
        return value.float()


def build(lock, device="cpu", geometry="native"):
    name, source = lock["model"], lock["source"]
    if name == "mobilenetv5":
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
    else:
        source_import("dinov3", source)
        from dinov3.hub.backbones import dinov3_vitl16

        model = dinov3_vitl16(pretrained=False)
        model.load_state_dict(
            convert_dino_hf(load_file(lock["checkpoint"]), model.state_dict()),
            strict=True,
        )
        transform = native_transforms(source).make_classification_eval_transform(
            resize_size=256, crop_size=224
        )
        encoder = Encoder(model, name, official_linear_input(source))
        common = TransformSpec()
    if geometry not in ("native", "letterbox224"):
        raise ValueError("Declare native or letterbox224 geometry")
    return encoder.to(device).eval(), (
        RGBTransform(transform) if geometry == "native" else common
    )
