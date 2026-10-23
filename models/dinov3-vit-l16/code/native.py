"""Released architectures, checked weight-format conversion and native readouts."""

import ast
import importlib
import importlib.util
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
        tokens = self.model.get_intermediate_layers(
            images, n=1, return_class_token=True, norm=True
        )
        value = self.readout(tokens, use_n_blocks=1, use_avgpool=True)
        if value.ndim != 2 or value.shape[1] != 2048 or not torch.isfinite(value).all():
            raise ValueError("Expected finite 2048-dimensional released features")
        return value.float()


def build(lock, device="cpu", geometry="native"):
    name, source = lock["model"], lock["source"]
    if name != 'dinov3':
        raise ValueError("This package only builds its declared encoder")
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
FAMILY='dinov3'
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

def load_assets(name, path):
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
