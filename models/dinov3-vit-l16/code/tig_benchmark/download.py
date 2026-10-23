"""Explicit asset download; uses HF's existing login, never accepts a CLI token."""

import json
from pathlib import Path
import tarfile
import urllib.request
from .assets import model_config, register


def fetch(name, output):

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
    weights_folder = Path(__file__).resolve().parents[2] / "weights"
    inventory = json.loads((weights_folder / "manifest.json").read_text())
    role = "best_encoder"
    relative = inventory["artifacts"][role]["path"]
    checkpoint = weights_folder / relative
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "model_release_weights", weights_folder / "download.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.download(weights_folder / "manifest.json", only=relative)
    return register(name, source, checkpoint, output / "assets.local.json")
