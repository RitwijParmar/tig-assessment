"""Image checks, split membership, local preparation and evaluation geometry."""

import csv, hashlib, io, json, re, zlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import atomic_json, digest, sha256

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("path", "group", "frame", "label6", "target", "crc", "original_bytes")


def checked_rows(rows):
    result, paths = [], set()
    for source in rows:
        row = dict(source)
        for key in ("path", "group"):
            row[key] = str(row[key])
            if not row[key]:
                raise ValueError("Missing image or parent identity")
        if row["path"] in paths:
            raise ValueError("Duplicate canonical image path")
        paths.add(row["path"])
        for key in ("target", "label6"):
            value = row[key]
            if str(value) != str(int(value)):
                raise ValueError("Labels must be exact integers")
            row[key] = int(value)
        if row["label6"] not in range(6) or row["target"] != int(row["label6"] > 0):
            raise ValueError("Released six-way/binary labels disagree")
        row["frame"] = int(row.get("frame", 0))
        result.append(row)
    if not result:
        raise ValueError("Empty canonical rows")
    return result


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


def preprocessing(config):
    value = dict(config["preprocessing"])
    for name in ("mean", "std", "padding_rgb"):
        value[name] = tuple(value[name])
    spec = TransformSpec(**value)
    if (
        spec.image_size != 224
        or spec.geometry != "letterbox"
        or spec.interpolation != "bilinear"
        or spec.mean != (0.485, 0.456, 0.406)
        or spec.std != (0.229, 0.224, 0.225)
        or spec.padding_rgb != (0, 0, 0)
    ):
        raise ValueError(
            "Declare whole-image letterbox224, bilinear, black padding and ImageNet normalization"
        )
    return spec


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


class ResearchImages(Dataset):
    def __init__(self, rows, data_root, transform):
        self.rows = [dict(r) for r in rows]
        self.root, self.transform = Path(data_root).resolve(), transform
        if not self.rows or len({r["path"] for r in self.rows}) != len(self.rows):
            raise ValueError("A nonempty unique declared image set is required")
        for row in self.rows:
            p = (self.root / row["path"]).resolve()
            if (
                self.root not in p.parents
                or not p.is_file()
                or p.stat().st_size != row["original_bytes"]
            ):
                raise ValueError(f"Missing or changed canonical image: {row['path']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = (self.root / row["path"]).read_bytes()
        if (
            len(raw) != row["original_bytes"]
            or zlib.crc32(raw) & 0xFFFFFFFF != row["crc"]
        ):
            raise ValueError(f"Canonical size/CRC changed: {row['path']}")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            original = torch.tensor(image.size, dtype=torch.long)
            tensor = self.transform(image)
        return {
            "image": tensor,
            "original_size": original,
            "target": torch.tensor(row["target"], dtype=torch.float32),
            "label6": row["label6"],
            "path": row["path"],
            "group": row["group"],
            "frame": row["frame"],
            "index": index,
        }


def normalize_rows(rows, data_root=None):
    normalized = []
    root = Path(data_root).resolve() if data_root else None
    for source in rows:
        row = dict(source)
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Frame paths must be relative and contained")
        group = re.sub(r"-part\d+$", "", path.parent.name)
        if row.get("group") and row["group"] != group:
            raise ValueError("Recording group disagrees with the filename-parent rule")
        row["group"] = group
        row["label6"] = int(row["label6"])
        target = int(row["label6"] > 0)
        if row.get("target") is not None and int(row["target"]) != target:
            raise ValueError("Binary target disagrees with released label")
        row["target"] = target
        row["frame"] = int(row.get("frame", re.search(r"(\d+)$", path.stem).group(1)))
        if row.get("crc") in (None, "") or row.get("original_bytes") in (None, ""):
            if root is None:
                raise ValueError("CRC/size or a local data root is required")
            file = (root / path).resolve()
            if root not in file.parents:
                raise ValueError("Resolved image escapes the data root")
            raw = file.read_bytes()
            row["crc"], row["original_bytes"] = zlib.crc32(raw) & 0xFFFFFFFF, len(raw)
        row["crc"], row["original_bytes"] = int(row["crc"]), int(row["original_bytes"])
        normalized.append({key: row[key] for key in FIELDS})
    return sorted(checked_rows(normalized), key=lambda row: row["path"])


def validate_roles(roles, fingerprints):
    expected = {"fitting", "selection", "assessment"}
    if set(roles) != expected:
        raise ValueError("Exactly fitting, selection and assessment roles required")
    group_sets = {role: {row["group"] for row in rows} for role, rows in roles.items()}
    all_paths = []
    for role, rows in roles.items():
        checked_rows(rows)
        all_paths.extend(row["path"] for row in rows)
        spec = fingerprints["roles"][role]
        hashes = sorted(
            hashlib.sha256(group.encode()).hexdigest() for group in group_sets[role]
        )
        if hashes != spec["group_sha256"] or len(rows) != spec["frames"]:
            raise ValueError("Recording/frame membership changed: " + role)
        if digest(rows) != spec["normalized_rows_sha256"]:
            raise ValueError("Frame identities, labels or checksums changed: " + role)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("A frame appears in multiple roles")
    if any(
        group_sets[a] & group_sets[b]
        for a, b in (
            ("fitting", "selection"),
            ("fitting", "assessment"),
            ("selection", "assessment"),
        )
    ):
        raise ValueError("A recording appears in multiple roles")
    return roles


def prepare(source, data_root, base_checkpoint, output):
    """Input is a local canonical CSV or the original local registration JSON."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Preparation requires a new output directory")
    if source.suffix.lower() == ".csv":
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = json.loads(source.read_text())["rows"]
    fingerprints = json.loads((ROOT / "config.json").read_text())["data_split"]
    role_for_hash = {
        value: role
        for role, spec in fingerprints["roles"].items()
        for value in spec["group_sha256"]
    }
    roles = {role: [] for role in fingerprints["roles"]}
    for row in normalize_rows(rows, data_root):
        role = role_for_hash.get(hashlib.sha256(row["group"].encode()).hexdigest())
        if role:
            roles[role].append(row)
    validate_roles(roles, fingerprints)
    output.mkdir(parents=True)
    manifest = output / "manifest.local.json"
    atomic_json(manifest, {"format": "ijepa-vit-h14-local-frames-v1", "roles": roles})
    config = json.loads((ROOT / "config.json").read_text())
    config.update(
        data_root=str(Path(data_root).resolve()),
        base_checkpoint_path=str(Path(base_checkpoint).resolve()),
        manifest_path=str(manifest.resolve()),
        manifest_sha256=sha256(manifest),
    )
    atomic_json(output / "config.local.json", config)
    return output / "config.local.json"


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("format") != "ijepa-vit-h14-native-v1":
        raise ValueError("Expected the standalone I-JEPA ViT-H/14 configuration")
    if (
        config.get("ssl_mask_backend") != "native"
        or config["image_size"] != 224
        or config["patch_size"] != 14
    ):
        raise ValueError(
            "The recorded final model uses native masking and H/14 at 224px"
        )
    if config["training"]["warmup_epochs"] != 1:
        raise ValueError("This recorded recipe uses one warmup epoch")
    for name in ("manifest_path", "data_root", "base_checkpoint_path"):
        if not Path(config[name]).is_absolute():
            raise ValueError("Prepare absolute local paths before training")
    if sha256(config["manifest_path"]) != config["manifest_sha256"]:
        raise ValueError("Local metadata manifest changed")
    return config


def load_roles(config):
    payload = json.loads(Path(config["manifest_path"]).read_text())
    roles = {role: normalize_rows(rows) for role, rows in payload["roles"].items()}
    fingerprints = json.loads((ROOT / "config.json").read_text())["data_split"]
    return validate_roles(roles, fingerprints)


def evaluation_loader(rows, config, batch_size=16):
    return DataLoader(
        ResearchImages(rows, config["data_root"], preprocessing(config)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["runtime"]["workers"],
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(config["seed"]),
    )
