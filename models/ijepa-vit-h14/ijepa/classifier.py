"""Train the final linear readout and load the selected encoder/head pair."""

from pathlib import Path
import gc, json
import torch
from torch import nn
from torch.nn import functional as F
from .utils import atomic_json, atomic_torch, digest, sha256
from .model import _strict_load
from .vendor.ijepa.vision_transformer import vit_huge
from .data import evaluation_loader, content_mask
from .metrics import class_weights, quality_metrics, selection_score

DEFAULT_RECIPE = json.loads(
    (Path(__file__).resolve().parents[1] / "config.json").read_text()
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


@torch.no_grad()
def extract(encoder, rows, config, device, batch_size=16):
    encoder.requires_grad_(False).eval()
    values = []
    seen = []
    for batch in evaluation_loader(rows, config, batch_size=batch_size):
        tokens = encoder(batch["image"].to(device)).float()
        valid = torch.stack([content_mask(s) for s in batch["original_size"]]).to(
            device
        )
        values.append(
            ((tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)).cpu()
        )
        seen.extend(batch["path"])
    if seen != [r["path"] for r in rows]:
        raise ValueError("Feature coverage/order mismatch")
    return torch.cat(values)


class WeldClassifier(nn.Module):
    def __init__(self, encoder, mean, std, head_state):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        if (
            mean.shape != std.shape
            or mean.ndim != 1
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("Invalid fitting-only feature normalization")
        self.head = nn.Linear(mean.numel(), 1)
        self.head.load_state_dict(head_state, strict=True)
        self.requires_grad_(False).eval()

    def forward(self, images, valid):
        tokens = self.encoder(images).float()
        pooled = (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        return self.head((pooled - self.mean) / self.std).squeeze(-1)

    @classmethod
    def from_files(cls, encoder_path, head_path, device="cpu", expected=None):
        if expected:
            for key, path in (("encoder", encoder_path), ("classifier", head_path)):
                if sha256(path) != expected[key]:
                    raise ValueError("Selected " + key + " checkpoint changed")
        exported = torch.load(
            encoder_path, map_location="cpu", weights_only=True, mmap=True
        )
        probe = torch.load(head_path, map_location="cpu", weights_only=True)
        with torch.device("meta"):
            encoder = vit_huge(img_size=[224], patch_size=14)
        _strict_load(encoder, exported["encoder"], "selected EMA encoder")
        return cls(encoder, probe["mean"], probe["std"], probe["head"]).to(device)

    @classmethod
    def from_run(cls, run, device="cpu"):
        run = Path(run)
        seal = json.loads((run / "seal.json").read_text())
        if seal["threshold"] != 0.5:
            raise ValueError("The recorded I-JEPA ViT-H/14 uses threshold 0.5")
        return cls.from_files(
            run / "encoder.pt", run / "classifier.pt", device, seal["checkpoint_sha256"]
        )
