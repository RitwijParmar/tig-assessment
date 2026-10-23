"""Pinned local source/weight registration; never silently download at train time."""

import hashlib
import json
from pathlib import Path
from .shared import ROOT, atomic_json, digest, sha256

MODEL_DIRS = {"dinov3": "dinov3-vit-l16"}


def model_config(name):
    if name not in MODEL_DIRS:
        raise ValueError("Unknown model")
    return json.loads((ROOT / "config.json").read_text())


def verify_source(name, folder):
    if name not in MODEL_DIRS:
        raise ValueError("Missing pinned source: undeclared model")
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


def register(name, source, checkpoint, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Asset registration already exists")
    config = model_config(name)
    revision = verify_source(name, source)
    actual = sha256(checkpoint)
    if actual != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint is not the declared released weight file")
    lock = {
        "format": "tig-model-assets-v1",
        "model": name,
        "source": str(Path(source).resolve()),
        "source_revision": revision,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": actual,
        "model_config_sha256": digest(config),
    }
    atomic_json(output, lock)
    return lock


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
