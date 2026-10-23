"""Audit model cards, shared data identities and optional local weight bytes.

Run from the repository root:
    python tools/verify_model_packages.py [--require-weights]
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = (
    "ijepa-vit-h14",
    "ijepa-vit-h14-pretrained",
    "mobilenetv5-300m",
    "dinov3-vit-l16",
)
ENTRIES = {"README.md", "model.json", "config.json", "results.json", "weights"}
SECTIONS = [
    "Status",
    "Architecture",
    "Data",
    "Training and selection",
    "Results",
    "Usage",
    "Weights",
    "Implementation and licenses",
]


def read(path):
    return json.loads(path.read_text())


def audit(require_weights=False):
    summary, identities, hashed = {}, [], {}
    for model_id in MODEL_IDS:
        folder = ROOT / "models" / model_id
        assert {p.name for p in folder.iterdir()} == ENTRIES, model_id
        package = read(folder / "model.json")
        config = read(folder / package["configuration"])
        result = read(folder / package["results"])
        inventory = read(folder / package["weights"])
        assert package["model_id"] == inventory["model_id"] == model_id
        assert package["format"] == "tig-model-package-v1"
        assert inventory["format"] == "tig-model-weights-v1"
        assert inventory["files_distributed_on_github"] is False
        assert (
            re.findall(r"^## (.+)$", (folder / "README.md").read_text(), re.M)
            == SECTIONS
        )
        assert (folder / package["implementation"]).is_dir()
        assert (
            result["threshold"]
            == package["architecture"]["threshold"]
            == config["classifier"]["threshold"]
            == 0.5
        )
        split = read(folder / package["data"]["split_config"])["data_split"]
        for role, values in package["data"]["roles"].items():
            assert values == {k: split["roles"][role][k] for k in values}, model_id
        identities.append(
            (
                set(package),
                set(package["architecture"]),
                set(package["training"]),
                package["data"],
                set(inventory["artifacts"]),
            )
        )
        checked = []
        available_names = set()
        for role, weight in inventory["artifacts"].items():
            if weight["path"] is None:
                assert weight["sha256"] is weight["bytes"] is None
                continue
            path = folder / "weights" / weight["path"]
            assert path.resolve().parent == (folder / "weights").resolve()
            assert re.fullmatch(r"[0-9a-f]{64}", weight["sha256"])
            assert weight["bytes"] > 0
            available_names.add(path.name)
            if require_weights:
                assert path.is_file() and not path.is_symlink(), path
                stat = path.stat()
                assert stat.st_size == weight["bytes"], path
                inode = (stat.st_dev, stat.st_ino)
                if inode not in hashed:
                    checksum = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 << 20), b""):
                            checksum.update(chunk)
                    hashed[inode] = checksum.hexdigest()
                assert hashed[inode] == weight["sha256"], path
                checked.append(role)
        assert {p.name for p in (folder / "weights").iterdir()} <= available_names | {
            "README.md",
            "manifest.json",
        }
        if package["status"] == "completed":
            assert result["validation"] is not None and result["assessment"] is not None
            assert (
                inventory["artifacts"]["classifier"]["sha256"]
                == result["weights"]["classifier_sha256"]
            )
        else:
            assert result["validation"] is result["assessment"] is None
            assert inventory["artifacts"]["classifier"]["path"] is None
        summary[model_id] = {
            "status": package["status"],
            "local_weight_roles_verified": checked,
        }
    assert all(
        x == identities[0] for x in identities
    ), "Model package schemas/data differ"
    for path in ROOT.rglob("*.md"):
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if re.match(r"\w+://", link) or link.startswith("#"):
                continue
            assert (path.parent / link.split("#")[0]).exists(), (path, link)
    return {
        "verified": True,
        "same_model_folder_structure": True,
        "same_model_card_sections": True,
        "same_package_metadata_schema": True,
        "exact_shared_data_identities": True,
        "relative_markdown_links_resolve": True,
        "local_weight_bytes_checked": require_weights,
        "unique_weight_files_hashed": len(hashed),
        "models": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(args.require_weights)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(content)
    print(content)
