"""Keep selected model weights independent of disposable download-cache paths."""

import os
from pathlib import Path
import shutil
import torch
from .shared import atomic_json, sha256


def retain(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Checkpoint export already exists")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    if sha256(source) != sha256(destination):
        raise ValueError("Retained checkpoint differs from source")
    return str(destination.resolve())


def export_selected(output, chosen, lock, config):
    output = Path(output)
    folder = output / "selected"
    base = retain(lock["checkpoint"], folder / "pretrained_encoder.safetensors")
    result = dict(chosen)
    result["head_path"] = retain(chosen["head_path"], folder / "classifier.pt")
    if chosen["encoder_path"]:
        result["encoder_path"] = retain(chosen["encoder_path"], folder / "encoder.pt")
    updated_assets = {**lock, "checkpoint": base}
    atomic_json(output / "assets.local.json", updated_assets)
    descriptor = {
        "model": config["model"],
        "mode": result["mode"],
        "geometry": config["geometry"],
        "threshold": 0.5,
        "selected_epoch": result["epoch"],
        "pretrained_encoder": {"path": base, "sha256": sha256(base)},
        "classifier": {
            "path": result["head_path"],
            "sha256": sha256(result["head_path"]),
        },
        "supervised_encoder": (
            {"path": result["encoder_path"], "sha256": sha256(result["encoder_path"])}
            if result["encoder_path"]
            else None
        ),
        "source_revision": lock.get("source_revision"),
        "note": "Selected final model; supervised final-epoch checkpoints are retained separately.",
    }
    atomic_json(folder / "model.json", descriptor)
    final_folder = output / "final"
    final_head = (
        output / "frozen/final.pt"
        if result["mode"] == "frozen"
        else Path(chosen["head_path"]).parent / "final_head.pt"
    )
    final_base = retain(base, final_folder / "pretrained_encoder.safetensors")
    final_classifier = retain(final_head, final_folder / "classifier.pt")
    final_encoder = None
    if result["mode"] == "supervised":
        final_encoder = retain(
            Path(chosen["encoder_path"]).parent / "final_encoder.pt",
            final_folder / "encoder.pt",
        )
    final_probe = torch.load(final_classifier, map_location="cpu", weights_only=True)
    final_descriptor = {
        **descriptor,
        "checkpoint_kind": "final_epoch_of_validation_selected_configuration",
        "final_epoch": final_probe["epoch"],
        "final_validation_score": final_probe["score"],
        "pretrained_encoder": {"path": final_base, "sha256": sha256(final_base)},
        "classifier": {"path": final_classifier, "sha256": sha256(final_classifier)},
        "supervised_encoder": (
            {"path": final_encoder, "sha256": sha256(final_encoder)}
            if final_encoder
            else None
        ),
        "note": "Last completed epoch of the validation-selected configuration; not selected using assessment.",
    }
    atomic_json(final_folder / "model.json", final_descriptor)
    return result
