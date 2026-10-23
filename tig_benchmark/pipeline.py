"""Fit on fitting rows, select on validation, seal before assessment."""

import hashlib
import json
import platform
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from . import models
from .assets import load_assets, model_config
from .data import load_data
from .shared import (
    ROOT,
    IJEPA_ROOT,
    ResearchImages,
    fit_probe,
    atomic_json,
    atomic_torch,
    digest,
    sha256,
    quality_metrics,
    configure_precision,
)


def implementation_identity():
    files = list((ROOT / "tig_benchmark").glob("*.py")) + list(
        (IJEPA_ROOT / "ijepa").rglob("*.py")
    )
    files += [ROOT / "tig_benchmark/upstream_sources.json", IJEPA_ROOT / "config.json"]
    return digest({str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)})


def tensor_identity(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def extract(encoder, transform, rows, config, device):
    encoder.eval().requires_grad_(False)
    loader = DataLoader(
        ResearchImages(rows, config["data_root"], transform),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
    )
    values, paths = [], []
    for batch in loader:
        values.append(encoder(batch["image"].to(device)).cpu())
        paths.extend(batch["path"])
    if paths != [r["path"] for r in rows]:
        raise ValueError("Feature frame coverage/order mismatch")
    return torch.cat(values)


def fit(
    name,
    data_path,
    asset_path,
    output,
    device="cpu",
    batch_size=None,
    workers=0,
    geometry="native",
    finetune=False,
):
    output, device = Path(output), torch.device(device)
    if output.exists():
        raise FileExistsError("Use a fresh run; never overwrite an existing selection")
    configure_precision()
    data, roles = load_data(data_path)
    lock = load_assets(name, asset_path)
    spec = model_config(name)
    if finetune and name != "mobilenetv5":
        raise ValueError("The declared supervised arm is MobileNet only")
    cfg = {
        **data,
        "model": name,
        "seed": 43,
        "workers": workers,
        "geometry": geometry,
        "batch_size": batch_size or spec["extraction_batch_size"],
        "device": str(device),
        "finetune": finetune,
        "recipe": spec["classifier"],
    }
    if cfg["batch_size"] < 1 or workers < 0:
        raise ValueError("Invalid batch/worker count")
    # Fail on missing fitting/validation images before allocating a huge model.
    for role in ("fitting", "selection"):
        ResearchImages(roles[role], cfg["data_root"], None)
    cfg["runtime"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "float_dtype": "float32",
        "matmul_precision": "highest",
        "convolution_precision": "tf32 on CUDA, matching recorded I-JEPA",
    }
    output.mkdir(parents=True)
    atomic_json(output / "config.json", cfg)
    atomic_json(output / "assets.local.json", lock)
    encoder, transform = models.build(lock, device, geometry)
    before = tensor_identity(encoder)
    features = {
        role: extract(encoder, transform, roles[role], cfg, device)
        for role in ("fitting", "selection")
    }
    if tensor_identity(encoder) != before:
        raise ValueError(
            "Frozen encoder weights/buffers changed during feature extraction"
        )
    atomic_torch(
        output / "features.local.pt",
        {
            "identity": digest({"assets": lock, "data": data, "geometry": geometry}),
            **features,
        },
    )
    probe = fit_probe(
        features["fitting"],
        features["selection"],
        roles["fitting"],
        roles["selection"],
        output / "frozen",
        name,
        spec["readout"],
        device,
        recipe=cfg["recipe"],
    )
    chosen = {
        "mode": "frozen",
        "epoch": probe["epoch"],
        "score": probe["score"],
        "head_path": str((output / "frozen/best.pt").resolve()),
        "encoder_path": None,
    }
    # Raw fitting-image fixture, never assessment images, for checkpoint replay.
    count = min(cfg["batch_size"], len(roles["fitting"]))
    dataset = ResearchImages(roles["fitting"][:count], cfg["data_root"], transform)
    images = torch.stack([dataset[i]["image"] for i in range(count)])
    atomic_torch(
        output / "replay.local.pt",
        {
            "identity": digest(roles["fitting"][:count]),
            "images": images,
            "features": features["fitting"][:count],
        },
    )
    del encoder, features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if finetune:
        from .finetune import run_candidate

        for rate in spec["supervised"]["encoder_lr_candidates"]:
            candidate = run_candidate(
                lock,
                roles,
                cfg,
                probe,
                rate,
                output / "supervised" / f"lr{rate:g}",
                device,
                spec["supervised"],
            )
            if candidate["score"]["selection_BA"] > chosen["score"]["selection_BA"]:
                chosen = {k: v for k, v in candidate.items() if k != "probe"}
                chosen["mode"] = "supervised"
    from .checkpoints import export_selected

    chosen = export_selected(output, chosen, lock, cfg)
    seal = {
        "format": "tig-transfer-selection-v1",
        "selected": chosen,
        "threshold": 0.5,
        "config_sha256": sha256(output / "config.json"),
        "assets_sha256": sha256(output / "assets.local.json"),
        "implementation_identity": implementation_identity(),
        "role_sha256": {k: digest(v) for k, v in roles.items()},
        "head_sha256": sha256(chosen["head_path"]),
        "replay_sha256": sha256(output / "replay.local.pt"),
        "encoder_sha256": (
            sha256(chosen["encoder_path"]) if chosen["encoder_path"] else None
        ),
        "frozen_encoder_tensor_sha256": before,
        "assessment_accessed_for_selection": False,
        "assessment_previously_inspected": True,
    }
    atomic_json(output / "seal.json", seal)
    verify(output, device)
    atomic_json(
        output / "status.json",
        {"stage": "sealed_and_reload_verified", "assessment_completed": False},
    )
    return chosen


def load_run(run, device):
    configure_precision()
    torch.set_default_dtype(torch.float32)
    run = Path(run)
    seal = json.loads((run / "seal.json").read_text())
    if seal.get("format") != "tig-transfer-selection-v1" or seal["threshold"] != 0.5:
        raise ValueError("Invalid selection seal")
    for filename, key in [
        ("config.json", "config_sha256"),
        ("assets.local.json", "assets_sha256"),
        ("replay.local.pt", "replay_sha256"),
    ]:
        if sha256(run / filename) != seal[key]:
            raise ValueError("Sealed run file changed: " + filename)
    if implementation_identity() != seal["implementation_identity"]:
        raise ValueError("Implementation changed after selection")
    cfg = json.loads((run / "config.json").read_text())
    _, roles = load_data_payload(cfg)
    if any(digest(v) != seal["role_sha256"][k] for k, v in roles.items()):
        raise ValueError("Recording roles changed")
    lock = load_assets(cfg["model"], run / "assets.local.json")
    chosen = seal["selected"]
    if sha256(chosen["head_path"]) != seal["head_sha256"]:
        raise ValueError("Selected head changed")
    encoder, transform = models.build(lock, device, cfg["geometry"])
    if chosen["encoder_path"]:
        if sha256(chosen["encoder_path"]) != seal["encoder_sha256"]:
            raise ValueError("Selected encoder changed")
        state = torch.load(
            chosen["encoder_path"], map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(state["encoder"], strict=True)
    encoder.eval().requires_grad_(False)
    probe = torch.load(chosen["head_path"], map_location="cpu", weights_only=True)
    if probe["fitting_rows_sha256"] != digest(roles["fitting"]) or probe[
        "selection_rows_sha256"
    ] != digest(roles["selection"]):
        raise ValueError("Head trained on different frame roles")
    return cfg, roles, encoder, transform, probe, seal


def load_data_payload(cfg):
    # Avoid a second writable data-config side channel; loader validates the
    # manifest digest and fixed fingerprints from the sealed run config itself.
    from .data import fingerprints
    from .shared import normalize_rows, validate_roles

    if sha256(cfg["manifest_path"]) != cfg["manifest_sha256"]:
        raise ValueError("Frame metadata changed after selection")
    rows = json.loads(Path(cfg["manifest_path"]).read_text())["roles"]
    roles = {k: normalize_rows(v) for k, v in rows.items()}
    return cfg, validate_roles(roles, fingerprints())


def head_logits(features, probe):
    head = nn.Linear(features.shape[1], 1)
    head.load_state_dict(probe["head"], strict=True)
    with torch.no_grad():
        return head((features.cpu() - probe["mean"]) / probe["std"]).squeeze(-1)


def verify(run, device="cpu"):
    cfg, roles, encoder, transform, probe, seal = load_run(run, device)
    frozen = seal["selected"]["encoder_path"] is None
    if frozen and tensor_identity(encoder) != seal["frozen_encoder_tensor_sha256"]:
        raise ValueError("Fresh encoder differs from fitted encoder")
    fixture = torch.load(
        Path(run) / "replay.local.pt", map_location="cpu", weights_only=True
    )
    with torch.no_grad():
        actual = encoder(fixture["images"].to(device)).cpu()
    if frozen and not torch.allclose(actual, fixture["features"], atol=1e-4, rtol=0):
        raise ValueError("Same-runtime raw-image feature reload discrepancy")
    # Also reproduce the selected supervised/frozen validation score after reload.
    values = extract(encoder, transform, roles["selection"], cfg, device)
    from .shared import selection_score

    score = selection_score(roles["selection"], head_logits(values, probe))
    if abs(score["selection_BA"] - seal["selected"]["score"]["selection_BA"]) > 1e-10:
        raise ValueError("Selected validation decisions did not reproduce")
    return {"verified": True, "selected_validation_BA": score["selection_BA"]}


def evaluate(run, device="cpu"):
    configure_precision()
    run = Path(run)
    if any(
        (run / f).exists()
        for f in (
            "assessment_metrics.json",
            "assessment_predictions.json",
            ".assessment.started",
        )
    ):
        raise FileExistsError(
            "Assessment already started; retain its original artifacts"
        )
    status_file = run / "status.json"
    if (
        not status_file.exists()
        or json.loads(status_file.read_text()).get("stage")
        != "sealed_and_reload_verified"
    ):
        raise ValueError("Training/reload verification must finish before assessment")
    cfg, roles, encoder, transform, probe, seal = load_run(run, device)
    # Exclusive creation prevents simultaneous evaluators and accidental repeats.
    (run / ".assessment.started").open("x").close()
    features = extract(encoder, transform, roles["assessment"], cfg, device)
    predictions, metrics = quality_metrics(
        roles["assessment"], head_logits(features, probe)
    )
    atomic_json(run / "assessment_predictions.json", predictions)
    atomic_json(run / "assessment_metrics.json", metrics)
    atomic_json(
        run / "status.json",
        {"stage": "assessment_completed", "assessment_completed": True},
    )
    return metrics["recording_weighted"]


def predict(run, image, device="cpu"):
    from PIL import Image

    cfg, _, encoder, transform, probe, _ = load_run(run, device)
    with Image.open(image) as im:
        tensor = transform(im.convert("RGB"))[None].to(device)
    with torch.no_grad():
        logit = float(head_logits(encoder(tensor).cpu(), probe)[0])
    return {
        "logit": logit,
        "defect_probability": float(torch.sigmoid(torch.tensor(logit))),
        "prediction": "defect" if logit >= 0 else "good",
        "threshold": 0.5,
    }
