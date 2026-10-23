"""Explicit asset download; uses HF's existing login, never accepts a CLI token."""

import json
from pathlib import Path
import tarfile
import urllib.request
from .assets import model_config, register, verify_source
from .shared import atomic_json, digest


def fetch(name, output, source_only=False):

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh asset directory")
    output.mkdir(parents=True)
    cfg = model_config(name)
    archive = output / "source.tar.gz"
    urllib.request.urlretrieve(
        f"https://codeload.github.com/{cfg['source_repository']}/tar.gz/{cfg['source_revision']}",
        archive,
    )
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        prefix = members[0].name.split("/")[0]
        if any(
            m.issym()
            or m.islnk()
            or m.name.startswith("/")
            or ".." in Path(m.name).parts
            for m in members
        ):
            raise ValueError("Unsafe upstream archive path")
        tar.extractall(output, filter="data")
    source = output / prefix
    archive.unlink()
    if source_only:
        lock = {
            "format": "tig-model-assets-v1", "model": name,
            "source": str(source.resolve()),
            "source_revision": verify_source(name, source),
            "checkpoint": None, "checkpoint_sha256": cfg["checkpoint_sha256"],
            "model_config_sha256": digest(cfg),
        }
        atomic_json(output / "assets.local.json", lock)
        return lock
    if name == "dinov3":
        from weights import download, read_manifest
        folder = Path(__file__).resolve().parents[2]
        inventory = read_manifest(folder)
        relative = inventory["artifacts"]["best_encoder"]["path"]
        download(folder / "weights/manifest.json", only=relative)
        checkpoint = folder / "weights" / relative
    else:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(
            repo_id=cfg["checkpoint_repository"], revision=cfg["checkpoint_revision"],
            filename="model.safetensors", local_dir=output / "training-weights",
        )
    return register(name, source, checkpoint, output / "assets.local.json")
