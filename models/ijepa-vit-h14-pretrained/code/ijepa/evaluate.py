"""Sealed assessment and single-image prediction for the selected I-JEPA model."""

import json
from pathlib import Path
import torch
from .utils import configure_precision, sha256, digest, atomic_json
from .data import (
    load_config,
    load_roles,
    content_mask,
    TransformSpec,
    evaluation_loader,
)
from .classifier import WeldClassifier
from .metrics import quality_metrics


def evaluate(run, device="cuda"):
    run = Path(run)
    if (run / "assessment_metrics.json").exists():
        raise FileExistsError("Assessment already exists; retain the original result")
    configure_precision()
    seal = json.loads((run / "seal.json").read_text())
    if sha256(run / "config.json") != seal["config_sha256"]:
        raise ValueError("Run configuration changed after selection")
    config = load_config(run / "config.json")
    if config["manifest_sha256"] != seal["manifest_sha256"]:
        raise ValueError("Frame metadata changed after selection")
    roles = load_roles(config)
    if any(digest(rows) != seal["role_sha256"][role] for role, rows in roles.items()):
        raise ValueError("Recording roles changed after selection")
    model = WeldClassifier.from_run(run, device)
    values, paths = [], []
    with torch.no_grad():
        for batch in evaluation_loader(roles["assessment"], config, batch_size=16):
            valid = torch.stack(
                [content_mask(size) for size in batch["original_size"]]
            ).to(device)
            values.append(model(batch["image"].to(device), valid).cpu())
            paths.extend(batch["path"])
    if paths != [row["path"] for row in roles["assessment"]]:
        raise ValueError("Assessment frame coverage/order mismatch")
    predictions, metrics = quality_metrics(roles["assessment"], torch.cat(values))
    atomic_json(run / "assessment_predictions.json", predictions)
    atomic_json(run / "assessment_metrics.json", metrics)
    return {
        key: metrics["recording_weighted"][key]
        for key in ("balanced_accuracy", "defect_recall", "false_alarm_rate")
    }


def predict_image(model, path, device="cpu"):
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        valid = content_mask(image.size)[None].to(device)
        tensor = TransformSpec()(image)[None].to(device)
    with torch.no_grad():
        logit = float(model(tensor, valid).cpu()[0])
    return {
        "image": str(path),
        "defect_probability": float(torch.sigmoid(torch.tensor(logit))),
        "logit": logit,
        "prediction": "defect" if logit >= 0 else "good",
        "threshold": 0.5,
    }
