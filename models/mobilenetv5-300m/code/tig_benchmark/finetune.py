"""Optional declared supervised MobileNet arm; epoch 0 remains eligible."""

import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .shared import (
    ResearchImages,
    class_weights,
    selection_score,
    atomic_json,
    atomic_torch,
    digest,
)
from .models import build


def schedule(step, total, warm):
    if step < warm:
        return (step + 1) / max(1, warm)
    return 0.5 * (
        1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm)))
    )


@torch.no_grad()
def logits_for(encoder, transform, probe, rows, config, device):
    head = nn.Linear(2048, 1).to(device)
    head.load_state_dict(probe["head"])
    mean, std = probe["mean"].to(device), probe["std"].to(device)
    loader = DataLoader(
        ResearchImages(rows, config["data_root"], transform),
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
    )
    encoder.eval()
    values = [
        head((encoder(b["image"].to(device)) - mean) / std).squeeze(-1).cpu()
        for b in loader
    ]
    return torch.cat(values)


def run_candidate(lock, roles, config, probe, rate, output, device, recipe):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config["seed"])
    encoder, transform = build(lock, device, config["geometry"])
    head = nn.Linear(2048, 1).to(device)
    head.load_state_dict(probe["head"])
    mean, std = probe["mean"].to(device), probe["std"].to(device)
    weights = class_weights(roles["fitting"]).to(device)
    from .pipeline import tensor_identity

    initial_encoder_identity = tensor_identity(encoder)
    encoder.requires_grad_(True)
    opt = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": rate},
            {"params": head.parameters(), "lr": recipe["head_lr"]},
        ],
        weight_decay=recipe["weight_decay"],
    )
    n = len(roles["fitting"])
    physical, effective = config["batch_size"], recipe["effective_batch"]
    if physical > effective or effective % physical:
        raise ValueError("Fine-tuning batch must divide effective batch 32")
    steps = math.ceil(n / effective)
    warm, total = recipe["warmup_epochs"] * steps, recipe["epochs"] * steps
    dataset = ResearchImages(roles["fitting"], config["data_root"], transform)
    best = {
        "epoch": 0,
        "score": dict(probe["score"]),
        "probe": probe,
        "encoder_path": None,
    }
    atomic_torch(output / "best_head.pt", probe)
    history = [{"epoch": 0, **probe["score"]}]
    stale, update, gradient_names = 0, 0, set()
    for epoch in range(1, recipe["epochs"] + 1):
        encoder.train()
        head.train()
        order = torch.randperm(
            n, generator=torch.Generator().manual_seed(config["seed"] + epoch)
        )
        groups = list(order.split(effective))
        batches, boundaries, masses = [], [], []
        for group in groups:
            chunks = list(group.split(physical))
            batches.extend([c.tolist() for c in chunks])
            masses.extend([len(group)] * len(chunks))
            boundaries.extend([False] * (len(chunks) - 1) + [True])
        loader = DataLoader(
            dataset, batch_sampler=batches, num_workers=config["workers"]
        )
        opt.zero_grad(set_to_none=True)
        for batch, end, mass in zip(loader, boundaries, masses):
            logits = head((encoder(batch["image"].to(device)) - mean) / std).squeeze(-1)
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, batch["target"].to(device), reduction="none"
                )
                * weights[batch["index"].to(device)]
            ).sum() / mass
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised loss")
            loss.backward()
            if end:
                gradient_names.update(
                    name for name, p in encoder.named_parameters() if p.grad is not None
                )
                if any(
                    not torch.isfinite(p.grad).all()
                    for p in list(encoder.parameters()) + list(head.parameters())
                    if p.grad is not None
                ):
                    raise ValueError("Nonfinite supervised gradients")
                scale = schedule(update, total, warm)
                for group, lr in zip(opt.param_groups, (rate, recipe["head_lr"])):
                    group["lr"] = lr * scale
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    recipe["gradient_clip"],
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                update += 1
        candidate = {
            **probe,
            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "identity": digest({"rate": rate, "epoch": epoch, "seed": config["seed"]}),
            "epoch": epoch,
        }
        vals = logits_for(
            encoder, transform, candidate, roles["selection"], config, device
        )
        score = selection_score(roles["selection"], vals)
        candidate["score"] = score
        history.append({"epoch": epoch, **score})
        if score["selection_BA"] > best["score"]["selection_BA"]:
            best = {
                "epoch": epoch,
                "score": score,
                "probe": candidate,
                "encoder_path": str((output / "best_encoder.pt").resolve()),
            }
            atomic_torch(
                output / "best_encoder.pt",
                {
                    "identity": candidate["identity"],
                    "encoder": {
                        k: v.detach().cpu() for k, v in encoder.state_dict().items()
                    },
                },
            )
            atomic_torch(output / "best_head.pt", candidate)
            stale = 0
        else:
            stale += 1
        if stale >= recipe["patience"]:
            break
    atomic_torch(
        output / "final_encoder.pt",
        {
            "identity": candidate["identity"],
            "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
        },
    )
    atomic_torch(output / "final_head.pt", candidate)
    atomic_json(output / "history.json", history)
    atomic_json(
        output / "weight_audit.json",
        {
            "parameters_with_gradients": sorted(gradient_names),
            "optimizer_updates": update,
            "warmup_updates": warm,
            "declared_schedule_updates": total,
            "epoch0_eligible": True,
            "selected_epoch": best["epoch"],
            "encoder_changed_during_supervised_training": tensor_identity(encoder)
            != initial_encoder_identity,
        },
    )
    return {**best, "head_path": str((output / "best_head.pt").resolve()), "rate": rate}
