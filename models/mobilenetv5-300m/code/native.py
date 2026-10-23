"""Released architectures, checked weight-format conversion and native readouts."""

import importlib
from pathlib import Path
import sys
import torch
from torch import nn
from safetensors.torch import load_file
from dataclasses import dataclass
import numpy as np
from PIL import Image


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


def build(lock, device="cpu", geometry="native", *, load_pretrained=True):
    name, source = lock["model"], lock["source"]
    if name != 'mobilenetv5':
        raise ValueError("This package only builds its declared encoder")
    timm = source_import("timm", source)
    model = timm.create_model("mobilenetv5_300m.gemma3n", pretrained=False)
    from timm.models.mobilenetv5 import checkpoint_filter_fn

    if load_pretrained:
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

import hashlib,json
from weights import read_manifest,checkpoint_file,sha256
ROOT=Path(__file__).resolve().parents[1]
FAMILY='mobilenetv5'
MODEL_DIRS={FAMILY:ROOT.name}

def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()

def model_config(name):
    if name not in MODEL_DIRS:
        raise ValueError("Unknown model")
    return json.loads((ROOT / "config.json").read_text())

def verify_source(name, folder):
    folder = Path(folder).resolve()
    key = "timm" if name == "mobilenetv5" else "dinov3"
    pin = json.loads((Path(__file__).parent / "upstream_sources.json").read_text())[key]
    for rel, expected in pin["files"].items():
        p = folder / rel
        if p.is_symlink() or not p.is_file():
            raise ValueError("Missing pinned source: " + rel)
        raw = p.read_bytes()
        if (
            hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            != expected
        ):
            raise ValueError("Modified pinned source: " + rel)
    # Extra importable modules in the model package could shadow pinned modules.
    package = folder / key
    for p in package.rglob("*.py"):
        if p.relative_to(folder).as_posix() not in pin["files"]:
            raise ValueError("Unexpected source module: " + str(p))
    return pin["revision"]

def load_assets(name, path, *, require_checkpoint=True):
    lock = json.loads(Path(path).read_text())
    config = model_config(name)
    if (
        lock.get("format") != "tig-model-assets-v1"
        or lock["model"] != name
        or lock["model_config_sha256"] != digest(config)
    ):
        raise ValueError("Assets/config identity mismatch")
    if verify_source(name, lock["source"]) != lock["source_revision"]:
        raise ValueError("Source revision changed")
    if lock["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint identity changed")
    if require_checkpoint:
        if not lock.get("checkpoint"):
            raise ValueError("Training requires a pretrained checkpoint")
        if sha256(lock["checkpoint"]) != config["checkpoint_sha256"]:
            raise ValueError("Pretrained checkpoint changed")
    return lock


def fetch_source(folder,output=None):
    import tarfile,urllib.request
    folder=Path(folder);output=Path(output or folder/'assets/source').resolve()
    if output.exists():raise FileExistsError('Use a fresh source directory')
    output.mkdir(parents=True)
    config=json.loads((folder/'config.json').read_text())
    archive=output/'source.tar.gz'
    urllib.request.urlretrieve(f"https://codeload.github.com/{config['source_repository']}/tar.gz/{config['source_revision']}",archive)
    with tarfile.open(archive) as tar:
        members=tar.getmembers();prefix=members[0].name.split('/')[0]
        if any(m.issym() or m.islnk() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe upstream archive path')
        tar.extractall(output,filter='data')
    archive.unlink();source=output/prefix
    revision=verify_source(FAMILY,source)
    inventory=read_manifest(folder)
    checkpoint=None
    if FAMILY=='dinov3':checkpoint=str(checkpoint_file(folder/'weights',inventory['artifacts']['best_encoder']).resolve())
    lock={'format':'tig-model-assets-v1','model':FAMILY,'source':str(source),'source_revision':revision,'checkpoint':checkpoint,'checkpoint_sha256':config['checkpoint_sha256'],'model_config_sha256':digest(config)}
    (output/'assets.local.json').write_text(json.dumps(lock,indent=2)+'\n')
    return lock
