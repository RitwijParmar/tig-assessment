"""I-JEPA checkpoint loading, preprocessing and inference."""
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
import json,numpy as np,torch
from torch import nn
from PIL import Image
from weights import read_manifest,checkpoint_file,sha256
from vendor.ijepa.vision_transformer import vit_huge

@dataclass(frozen=True)
class TransformSpec:
    """RGB conversion, centered whole-image letterbox, and fixed normalization."""

    image_size: int = 224
    geometry: str = "letterbox"
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"
    padding_rgb: tuple = (0, 0, 0)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Transform requires a PIL image; RGB conversion is explicit"
            )
        image = image.convert("RGB")
        size = self.image_size
        resample = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }[self.interpolation]
        if self.geometry == "letterbox":
            scale = size / max(image.size)
            width, height = [max(1, round(value * scale)) for value in image.size]
            canvas = Image.new("RGB", (size, size), self.padding_rgb)
            canvas.paste(
                image.resize((width, height), resample),
                ((size - width) // 2, (size - height) // 2),
            )
            image = canvas
        elif self.geometry == "stretch":
            image = image.resize((size, size), resample)
        else:
            raise ValueError("geometry must be letterbox or stretch")
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        pixels = pixels.permute(2, 0, 1) / 255.0
        return (
            pixels - pixels.new_tensor(self.mean)[:, None, None]
        ) / pixels.new_tensor(self.std)[:, None, None]

def content_mask(original_size, image_size=224, patch_size=14):
    """True only for 14px patches wholly inside the letterboxed image."""
    if isinstance(original_size, torch.Tensor):
        original_size = original_size.cpu().tolist()
    if len(original_size) != 2 or any(int(v) != v or v <= 0 for v in original_size):
        raise ValueError("Positive width,height required")
    width, height = map(int, original_size)
    scale = image_size / max(width, height)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (image_size - width) // 2, (image_size - height) // 2
    starts = torch.arange(image_size // patch_size) * patch_size
    columns = (starts >= left) & (starts + patch_size <= left + width)
    rows = (starts >= top) & (starts + patch_size <= top + height)
    return (rows[:, None] & columns[None, :]).flatten()

def _strict_load(module, raw, role):
    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint has no tensor dictionary for {role}")
    cleaned = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Non-tensor checkpoint state in {role}")
        name = key
        while name.startswith("module."):
            name = name[len("module.") :]
        if name in cleaned:
            raise ValueError(f"Colliding normalized {role} key: {name}")
        cleaned[name] = value
    expected = module.state_dict()
    missing = sorted(expected.keys() - cleaned.keys())
    unexpected = sorted(cleaned.keys() - expected.keys())
    wrong_shape = {
        name: {
            "expected": list(expected[name].shape),
            "actual": list(cleaned[name].shape),
        }
        for name in expected.keys() & cleaned.keys()
        if expected[name].shape != cleaned[name].shape
    }
    if missing or unexpected or wrong_shape:
        raise ValueError(
            f"Strict {role} state audit failed: "
            + json.dumps(
                {
                    "missing": missing,
                    "unexpected": unexpected,
                    "shape_errors": wrong_shape,
                }
            )
        )
    module.load_state_dict(cleaned, strict=True, assign=True)
    return {
        "role": role,
        "strict": True,
        "tensor_count": len(cleaned),
        "missing": [],
        "unexpected": [],
        "checkpoint_dtypes": sorted({str(value.dtype) for value in cleaned.values()}),
    }

def configure_precision():
    """Match the recorded float32/highest-matmul/default-TF32-convolution recipe."""
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cudnn, "conv"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        torch.backends.cudnn.allow_tf32 = True

class WeldClassifier(nn.Module):
    def __init__(self, encoder, mean, std, head_state):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        if (
            mean.shape != std.shape
            or mean.ndim != 1
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("Invalid fitting-only feature normalization")
        self.head = nn.Linear(mean.numel(), 1)
        self.head.load_state_dict(head_state, strict=True)
        self.requires_grad_(False).eval()

    def forward(self, images, valid):
        tokens = self.encoder(images).float()
        pooled = (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        return self.head((pooled - self.mean) / self.std).squeeze(-1)

    @classmethod
    def from_files(
        cls,
        encoder_path,
        head_path,
        device="cpu",
        expected=None,
        encoder_role="encoder",
    ):
        if encoder_role not in {"encoder", "target_encoder"}:
            raise ValueError("Use an exported encoder or the original target_encoder")
        if expected:
            for key, path in (("encoder", encoder_path), ("classifier", head_path)):
                if sha256(path) != expected[key]:
                    raise ValueError("Selected " + key + " checkpoint changed")
        exported = torch.load(
            encoder_path, map_location="cpu", weights_only=True, mmap=True
        )
        probe = torch.load(head_path, map_location="cpu", weights_only=True)
        with torch.device("meta"):
            encoder = vit_huge(img_size=[224], patch_size=14)
        _strict_load(encoder, exported[encoder_role], encoder_role)
        return cls(encoder, probe["mean"], probe["std"], probe["head"]).to(device)

def predict_image(model, path, device="cpu"):
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        valid = content_mask(image.size)[None].to(device)
        tensor = TransformSpec()(image)[None].to(device)
    with torch.no_grad():
        logit = float(model(tensor, valid).cpu()[0])
    return {
        "image": str(path),
        "defect_probability": float(torch.sigmoid(torch.tensor(logit))),
        "logit": logit,
        "prediction": "defect" if logit >= 0 else "good",
        "threshold": 0.5,
    }

def predict(folder,image,device='cpu',assets=None):
    folder=Path(folder);inventory=read_manifest(folder);configure_precision()
    encoder=checkpoint_file(folder/'weights',inventory['artifacts']['best_encoder'])
    head=checkpoint_file(folder/'weights',inventory['artifacts']['best_classifier'])
    model=WeldClassifier.from_files(encoder,head,device)
    return predict_image(model,image,device)
