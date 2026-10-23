"""Unchanged I-JEPA linear-probe fitting function; architecture independent."""

from pathlib import Path
import gc
import torch
from torch import nn
from torch.nn import functional as F
from .utils import atomic_json, atomic_torch, digest
from .metrics import class_weights, quality_metrics, selection_score

DEFAULT_RECIPE = __import__("json").loads(
    (Path(__file__).resolve().parents[2] / "config.json").read_text()
)["classifier"]


def fit_probe(
    train_features,
    val_features,
    fitting,
    selection,
    output,
    candidate,
    readout,
    device,
    recipe=None,
):
    recipe = dict(recipe or DEFAULT_RECIPE)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"]) if device.type == "cuda" else None
    # CPU population moments shared across every recipe, never validation moments.
    mean = train_features.mean(0)
    std = train_features.std(0, unbiased=False).clamp_min(1e-6)
    x = ((train_features - mean) / std).to(device)
    val = ((val_features - mean) / std).to(device)
    y = torch.tensor([r["target"] for r in fitting], device=device, dtype=torch.float32)
    weights = class_weights(fitting).to(device)
    head = nn.Linear(x.shape[1], 1, device=device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(
        head.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )
    best = -1.0
    stale = 0
    history = []
    saved = None
    for epoch in range(1, recipe["epochs"] + 1):
        head.train()
        total = 0.0
        order = torch.randperm(
            len(fitting),
            generator=torch.Generator().manual_seed(recipe["seed"] + epoch),
        )
        for idx_cpu in order.split(recipe["batch_size"]):
            idx = idx_cpu.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (
                F.binary_cross_entropy_with_logits(
                    head(x[idx]).squeeze(-1), y[idx], reduction="none"
                )
                * weights[idx]
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite probe loss")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        head.eval()
        with torch.no_grad():
            logits = head(val).squeeze(-1)
        score = selection_score(selection, logits)
        history.append({"epoch": epoch, "fitting_loss": total / len(fitting), **score})
        if score["selection_BA"] > best:
            best = score["selection_BA"]
            stale = 0
            saved = {
                "identity": digest(
                    {"candidate": candidate, "readout": readout, "recipe": recipe}
                ),
                "candidate": candidate,
                "readout": readout,
                "recipe": recipe,
                "head": {
                    k: v.detach().cpu().clone() for k, v in head.state_dict().items()
                },
                "mean": mean,
                "std": std,
                "epoch": epoch,
                "fitting_rows_sha256": digest(fitting),
                "selection_rows_sha256": digest(selection),
                "score": score,
            }
        else:
            stale += 1
        if stale >= recipe["patience"]:
            break
    final = {
        **saved,
        "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        "epoch": epoch,
        "score": score,
        "checkpoint_kind": "final_epoch",
    }
    atomic_torch(output / "final.pt", final)
    atomic_torch(output / "best.pt", saved)
    atomic_json(output / "history.json", history)
    # Check selected CPU reloaded head and full metrics agree with selection.
    h = nn.Linear(val_features.shape[1], 1)
    h.load_state_dict(saved["head"])
    h.eval()
    with torch.no_grad():
        logits = h((val_features - saved["mean"]) / saved["std"]).squeeze(-1)
    preds, metrics = quality_metrics(selection, logits)
    if abs(metrics["recording_weighted_balanced_accuracy"] - best) > 1e-10:
        raise ValueError("GPU/CPU selection decision discrepancy")
    atomic_json(output / "selection_metrics.json", metrics)
    atomic_json(output / "selection_predictions.json", preds)
    del x, val, head, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return saved
