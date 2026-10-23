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


# Some pretrained RMS-normalized paths yield ~1e38 stem derivatives on dark
# fitting images. Scale the scalar loss down during backward, then unscale and
# apply the original global norm cap in one numerically stable operation.
BACKWARD_SCALE = 2.0 ** -64


def clip_scaled_gradients(parameters, max_norm, scale=BACKWARD_SCALE):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        raise ValueError("No gradients to clip")
    norms = torch.stack([
        torch.linalg.vector_norm(g, dtype=torch.float64) for g in grads
    ])
    scaled_norm = float(torch.linalg.vector_norm(norms))
    if not math.isfinite(scaled_norm):
        raise ValueError("Nonfinite scaled gradient norm")
    # Equivalent to unscale followed by clip_grad_norm_(..., max_norm),
    # including its 1e-6 stabilizer, without materializing overflowing norms.
    factor = min(1.0 / scale, max_norm / (scaled_norm + 1e-6 * scale))
    for grad in grads:
        grad.mul_(factor)
    return scaled_norm / scale


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


def run_candidate(lock, roles, config, probe, rate, output, device, recipe, resume=False):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError("Use resume only for a preserved unfinished candidate")
    if output.exists() and not (output / "resume.private.pt").is_file():
        raise ValueError("Candidate exists without a complete epoch checkpoint")
    output.mkdir(parents=True, exist_ok=True)
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
    history = [{"epoch": 0, **probe["score"]}]
    stale, update, gradient_names = 0, 0, set()
    start_epoch = 1
    checkpoint = output / "resume.private.pt"
    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = digest({"config": config, "recipe": recipe, "rate": rate})
        if state["identity"] != expected or state["backward_loss_scale"] != BACKWARD_SCALE:
            raise ValueError("Resume configuration/scale differs")
        encoder.load_state_dict(state["encoder"], strict=True)
        head.load_state_dict(state["head"], strict=True)
        opt.load_state_dict(state["optimizer"])
        history, stale, update = state["history"], state["stale"], state["updates"]
        start_epoch = state["completed_epoch"] + 1
        best_probe = torch.load(output / "best_head.pt", map_location="cpu", weights_only=True)
        expected_epoch = state["best_epoch"] if state["best_epoch"] else probe["epoch"]
        if best_probe["epoch"] != expected_epoch or best_probe["score"] != state["best_score"]:
            raise ValueError("Best checkpoint is not at the optimizer commit boundary; preserve files and finish/recover that checkpoint transaction")
        if state["best_epoch"] > 0:
            retained_best = torch.load(output / "best_encoder.pt", map_location="cpu", weights_only=True)
            if retained_best["identity"] != best_probe["identity"]:
                raise ValueError("Best encoder/head commit differs")
            del retained_best
        best = dict(epoch=state["best_epoch"], score=state["best_score"], probe=best_probe,
            encoder_path=str((output / "best_encoder.pt").resolve()) if state["best_epoch"] > 0 else None)
        gradient_names = {name for name, param in encoder.named_parameters() if param in opt.state}
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        completed = state["completed_epoch"] >= recipe["epochs"] or stale >= recipe["patience"]
        del state
        if completed:
            if not (output / "weight_audit.json").exists():
                atomic_json(output / "weight_audit.json", {
                    "parameters_with_gradients": sorted(gradient_names), "optimizer_updates": update,
                    "warmup_updates": warm, "declared_schedule_updates": total,
                    "epoch0_eligible": True, "selected_epoch": best["epoch"],
                    "encoder_changed_during_supervised_training": tensor_identity(encoder) != initial_encoder_identity,
                    "backward_loss_scale": BACKWARD_SCALE, "gradient_norm_accumulation_dtype": "float64",
                    "gradient_clip": recipe["gradient_clip"],
                })
            return {**best, "head_path": str((output / "best_head.pt").resolve()), "rate": rate}
    else:
        atomic_torch(output / "best_head.pt", probe)
    for epoch in range(start_epoch, recipe["epochs"] + 1):
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
            (loss * BACKWARD_SCALE).backward()
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
                clip_scaled_gradients(
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
        # Durable epoch boundaries retain the actual final weights even if a
        # spot VM stops before the next epoch. Selection still uses validation.
        atomic_torch(
            output / "final_encoder.pt",
            {
                "identity": candidate["identity"],
                "encoder": {
                    k: v.detach().cpu() for k, v in encoder.state_dict().items()
                },
            },
        )
        atomic_torch(output / "final_head.pt", candidate)
        atomic_json(output / "history.json", history)
        atomic_torch(
            output / "resume.private.pt",
            {
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "completed_epoch": epoch,
                "backward_loss_scale": BACKWARD_SCALE,
                "updates": update,
                "stale": stale,
                "history": history,
                "best_epoch": best["epoch"],
                "best_score": best["score"],
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "identity": digest({"config": config, "recipe": recipe, "rate": rate}),
            },
        )
        print(
            {
                "stage": "supervised_epoch",
                "lr": rate,
                "epoch": epoch,
                "validation_BA": score["selection_BA"],
                "best_epoch": best["epoch"],
            },
            flush=True,
        )
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
            "backward_loss_scale": BACKWARD_SCALE,
            "gradient_norm_accumulation_dtype": "float64",
            "gradient_clip": recipe["gradient_clip"],
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
