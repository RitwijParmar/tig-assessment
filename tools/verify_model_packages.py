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
    "ijepa-vit-h14-adapted",
    "ijepa-vit-h14-pretrained",
    "mobilenetv5-300m",
    "dinov3-vit-l16",
)
ENTRIES = {
    "README.md",
    "model.json",
    "config.json",
    "data.json",
    "results.json",
    "requirements.txt",
    "run.py",
    "code",
    "tests",
    "weights",
    "licenses",
}
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
        assert {
            p.name
            for p in folder.iterdir()
            if p.name not in ("__pycache__", ".pytest_cache", "assets", "runs")
        } == ENTRIES, model_id
        package = read(folder / "model.json")
        config = read(folder / package["configuration"])
        result = read(folder / package["results"])
        inventory = read(folder / package["weights"])
        assert package["model_id"] == inventory["model_id"] == model_id
        assert package["format"] == "tig-model-package-v1"
        assert inventory["format"] == "tig-model-weights-v1"
        assert inventory["files_distributed_on_github"] is True
        assert package["status"] == result["status"] == "completed"
        assert "selected_final_model" not in package and "selected_final_model" not in result
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
        split = read(folder / package["data"]["split_config"])
        for role, values in package["data"]["roles"].items():
            assert values == {k: split["roles"][role][k] for k in values}, model_id
        identities.append(
            (
                set(package),
                set(package["architecture"]),
                set(package["training"]),
                package["data"],
            )
        )
        checked = []
        available_names = set()
        paths = set()
        for role, weight in inventory["artifacts"].items():
            assert weight["path"] and weight["required"] is True
            assert weight["path"] not in paths, "One file must not have duplicate inventory entries"
            paths.add(weight["path"])
            path = folder / "weights" / weight["path"]
            assert path.resolve().is_relative_to((folder / "weights").resolve())
            assert ".." not in Path(weight["path"]).parts
            assert re.fullmatch(r"[0-9a-f]{64}", weight["sha256"])
            assert weight["bytes"] > 0
            available_names.add(Path(weight["path"]).parts[0])
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
        assert {
            p.name for p in (folder / "weights").iterdir() if p.name != "__pycache__"
        } <= available_names | {
            "README.md",
            "manifest.json",
            "download.py",
        }
        pairs = inventory["checkpoints"]
        assert set(inventory["artifacts"]) == {"best_encoder", "best_classifier", "final_encoder", "final_classifier"}
        assert available_names == {"best", "final"}
        allowed_weight_paths = {
            weight["path"] for weight in inventory["artifacts"].values()
            if weight["path"] is not None
        } | {"README.md", "manifest.json", "download.py"} | {p["descriptor"] for p in pairs.values()}
        for weight_file in (folder / "weights").rglob("*"):
            if "__pycache__" in weight_file.parts or not weight_file.is_file():
                continue
            assert weight_file.relative_to(folder / "weights").as_posix() in allowed_weight_paths, weight_file
        assert set(inventory["checkpoints"]) == {"best", "final"}
        assert all(inventory["artifacts"][inventory["checkpoints"]["best"][key]]["path"].startswith("best/") for key in ("encoder", "classifier"))
        for kind, pair in pairs.items():
            assert kind in ("best", "final")
            assert all(inventory["artifacts"][pair[key]]["path"].startswith(kind+"/") for key in ("encoder", "classifier"))
            descriptor_path = folder / "weights" / kind / "model.json"
            assert descriptor_path.is_file(), descriptor_path
            if descriptor_path.exists():
                descriptor = read(descriptor_path)
                assert descriptor["format"] == "tig-checkpoint-v1"
                assert descriptor["model_id"] == model_id and descriptor["kind"] == kind
                assert descriptor["threshold"] == 0.5
                assert descriptor["assessment_used_to_select"] is False
                assert descriptor["source_model_config_sha256"] == hashlib.sha256(
                    (folder / "config.json").read_bytes()
                ).hexdigest()
                for role in ("encoder", "classifier", "pretrained_encoder"):
                    item = descriptor.get(role)
                    if not item:
                        continue
                    relative = item["path"]
                    assert ".." not in Path(relative).parts and not Path(relative).is_absolute()
                    assert any(
                        weight["path"] == relative and weight["sha256"] == item["sha256"]
                        for weight in inventory["artifacts"].values()
                    ), (model_id, relative)
                for role in ("encoder", "classifier"):
                    assert descriptor[role]["path"] == inventory["artifacts"][pair[role]]["path"]
        if "final" not in inventory["checkpoints"]:
            assert not (folder / "weights/final").exists()
        if package["status"] == "completed":
            assert result["validation"] is not None and result["assessment"] is not None
            assert (
                inventory["artifacts"]["best_classifier"]["sha256"]
                == result["weights"]["classifier_sha256"]
            )
        else:
            assert result["validation"] is result["assessment"] is None
            assert inventory["artifacts"]["best_classifier"]["path"] is None
        summary[model_id] = {
            "status": package["status"],
            "local_weight_roles_verified": checked,
        }
    assert all(
        x == identities[0] for x in identities
    ), "Model package schemas/data differ"
    assert len(re.findall(r"\| Completed \|", (ROOT / "README.md").read_text())) == len(MODEL_IDS)
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
