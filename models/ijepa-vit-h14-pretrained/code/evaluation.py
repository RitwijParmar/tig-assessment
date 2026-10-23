"""Evaluate the published best pair without fitting or selecting anything."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from inference import load_best
from weights import read_manifest


def evaluate_best(folder, data, output, device="cpu", assets=None, batch_size=2):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh assessment directory; preserve earlier results")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    inventory = read_manifest(folder)
    ijepa = inventory["inference"]["family"] == "ijepa"
    if ijepa:
        from ijepa.data import load_config, load_roles, preprocessing, ResearchImages, content_mask
        from ijepa.metrics import quality_metrics
        from ijepa.utils import atomic_json, sha256
        config = load_config(data)
        roles = load_roles(config)
        transform = preprocessing(config)
    else:
        from tig_benchmark.data import load_data
        from tig_benchmark.shared import ResearchImages, quality_metrics, atomic_json, sha256
        from tig_benchmark.pipeline import head_logits
        config, roles = load_data(data)
    # Registration verifies the exact frame memberships and labels before loading.
    rows = roles["assessment"]
    if ijepa:
        model = load_best(folder, device, assets)
    else:
        encoder, transform, probe, descriptor = load_best(folder, device, assets)
    loader = DataLoader(ResearchImages(rows, config["data_root"], transform),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    output.mkdir(parents=True)
    atomic_json(output / "evaluation.json", {
        "format": "tig-best-assessment-v1", "model": inventory["model_id"],
        "weights": {k: v["sha256"] for k, v in inventory["artifacts"].items()},
        "prepared_data_sha256": sha256(data), "threshold": 0.5,
        "assessment_previously_inspected": True,
        "normalization": "loaded from best classifier; never refitted",
    })
    values, paths = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if ijepa:
                valid = torch.stack([content_mask(s) for s in batch["original_size"]]).to(device)
                values.append(model(images, valid).cpu())
            else:
                values.append(head_logits(encoder(images).cpu(), probe))
            paths.extend(batch["path"])
    if paths != [row["path"] for row in rows]:
        raise ValueError("Assessment frame coverage/order mismatch")
    predictions, metrics = quality_metrics(rows, torch.cat(values))
    atomic_json(output / "assessment_predictions.json", predictions)
    atomic_json(output / "assessment_metrics.json", metrics)
    return metrics["recording_weighted"]
