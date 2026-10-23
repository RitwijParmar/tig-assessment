"""Exact final I-JEPA frame roles; no random split or validation refit."""

import csv
import hashlib
import json
from pathlib import Path
from .shared import IJEPA_ROOT, normalize_rows, validate_roles, atomic_json, sha256


def fingerprints():
    return json.loads((IJEPA_ROOT / "config.json").read_text())["data_split"]


def prepare(metadata, data_root, output):
    output, metadata = Path(output), Path(metadata)
    if output.exists():
        raise FileExistsError("Use a fresh preparation directory")
    if metadata.suffix.lower() == ".csv":
        with metadata.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = json.loads(metadata.read_text())["rows"]
    spec = fingerprints()
    role_hash = {
        h: role for role, v in spec["roles"].items() for h in v["group_sha256"]
    }
    roles = {role: [] for role in spec["roles"]}
    for row in normalize_rows(rows, data_root):
        role = role_hash.get(hashlib.sha256(row["group"].encode()).hexdigest())
        if role:
            roles[role].append(row)
    validate_roles(roles, spec)
    output.mkdir(parents=True)
    manifest = output.resolve() / "manifest.local.json"
    atomic_json(manifest, {"format": "tig-matched-frames-v1", "roles": roles})
    config = {
        "format": "tig-matched-data-v1",
        "data_root": str(Path(data_root).resolve()),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256(manifest),
    }
    atomic_json(output / "data.local.json", config)
    return config


def load_data(path):
    config = json.loads(Path(path).read_text())
    if config.get("format") != "tig-matched-data-v1":
        raise ValueError("Use the matched data preparation command")
    if not all(Path(config[k]).is_absolute() for k in ("data_root", "manifest_path")):
        raise ValueError("Absolute prepared paths required")
    if sha256(config["manifest_path"]) != config["manifest_sha256"]:
        raise ValueError("Frame manifest changed")
    raw = json.loads(Path(config["manifest_path"]).read_text())["roles"]
    roles = {role: normalize_rows(rows) for role, rows in raw.items()}
    validate_roles(roles, fingerprints())
    return config, roles
