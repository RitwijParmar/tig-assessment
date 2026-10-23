"""Explicit asset download; uses HF's existing login, never accepts a CLI token."""

from pathlib import Path
import tarfile
import urllib.request
from .assets import model_config, register


def fetch(name, output):
    from huggingface_hub import hf_hub_download

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
    checkpoint = hf_hub_download(
        cfg["checkpoint_repository"],
        "model.safetensors",
        revision=cfg["checkpoint_revision"],
        cache_dir=output / "cache",
    )
    return register(name, source, checkpoint, output / "assets.local.json")
