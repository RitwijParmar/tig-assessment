"""I-JEPA ViT-H/14 training: native JEPA adaptation followed by linear readouts."""

from collections import Counter
import gc, hashlib, importlib.util, json, math, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .data import ResearchImages, preprocessing, load_config, load_roles
from .utils import (
    atomic_json,
    atomic_torch,
    digest,
    rng_state,
    restore_rng,
    sha256,
    exclusive_run,
    configure_precision,
)
from .model import build_model
from .classifier import extract, fit_probe

UPSTREAM = Path(__file__).resolve().parent / "vendor/ijepa"
PINNED = {
    "transforms.py": "441acaef22ca3bd121f98f95bfcad2b271ecacef0ab4ba72d4f3dc28c57fd44c",
    "multiblock.py": "a06c593636f81e760bd26b615b2a7a976790dcc027d4b9244f3e8e7a2e5a8804",
}


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _schedule(step, total_steps, warm_steps):
    if step < warm_steps:
        return (step + 1) / max(1, warm_steps)
    return 0.5 * (
        1
        + math.cos(
            math.pi * min(1.0, (step - warm_steps) / max(1, total_steps - warm_steps))
        )
    )


def upstream(name):
    path = UPSTREAM / name
    if name not in PINNED or sha256(path) != PINNED[name]:
        raise ValueError("Pinned upstream source changed: " + name)
    spec = importlib.util.spec_from_file_location("ijepa_native_" + name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def view_seed(seed, identity, epoch, view):
    return int.from_bytes(
        hashlib.sha256(json.dumps([seed, identity, epoch, view]).encode()).digest()[:8],
        "big",
    ) % (2**63 - 1)


class NativeRGBTransform:
    def __init__(self):
        self.transform = upstream("transforms.py").make_transforms(
            crop_size=224,
            crop_scale=(0.3, 1.0),
            horizontal_flip=False,
            color_distortion=False,
            gaussian_blur=False,
        )

    def __call__(self, image):
        # Official ImageFolder's PIL loader converts every source to RGB before
        # make_transforms. Our CRC-checked loader must make the same conversion.
        return self.transform(image.convert("RGB"))


class SeededImages(Dataset):
    """Native crop randomized per image/epoch, independent of worker scheduling."""

    def __init__(self, rows, config, epoch, view, random_crop):
        transform = NativeRGBTransform() if random_crop else preprocessing(config)
        self.dataset = ResearchImages(rows, config["data_root"], transform)
        self.seed, self.epoch, self.view = config["seed"], epoch, view
        self.random_crop = random_crop

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.rows[index]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                view_seed(self.seed, row["path"], self.epoch, self.view)
            )
            result = self.dataset[index]
        # Masks operate in the geometry after the crop; no letterbox padding.
        result["mask_size"] = (
            torch.tensor([224, 224]) if self.random_crop else result["original_size"]
        )
        return result


def loader(rows, config, epoch, view=0, training=False, batch_size=None, ssl=False):
    dataset = SeededImages(rows, config, epoch, view, ssl)
    return DataLoader(
        dataset,
        batch_size=batch_size or config["training"]["batch_size"],
        shuffle=training,
        num_workers=config["runtime"]["workers"],
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(config["seed"] + epoch),
    )


class NativeMasker:
    """Calls the exact official collator, including its documented retry fallback."""

    def __init__(self, seed=43):
        self.seed = seed
        self.collator = upstream("multiblock.py").MaskCollator(
            input_size=(224, 224),
            patch_size=14,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
            nenc=1,
            npred=4,
            min_keep=10,
            allow_overlap=False,
        )

    def sample(self, ids, sizes=None, epoch=0, view=0, device=None):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                view_seed(self.seed, list(ids), epoch, view)
            )
            _, contexts, targets = self.collator([{"id": i} for i in ids])
        context = contexts[0]
        overlap = sum(
            int((context[:, :, None] == t[:, None, :]).any(-1).sum()) for t in targets
        )
        return {
            "context_indices": context.to(device),
            "target_indices": [t.to(device) for t in targets],
            "context_tokens": context.shape[1],
            "target_tokens_per_block": targets[0].shape[1],
            "native_overlap_context_target_pairs": overlap,
            "valid_tokens": [256] * len(ids),
            "upstream_sha256": PINNED["multiblock.py"],
        }


def ssl_loss(model, masks, batch, epoch, view, device):
    drawn = masks.sample(
        batch["path"], batch["mask_size"], epoch=epoch, view=view, device=device
    )
    values = model(
        batch["image"].to(device), drawn["context_indices"], drawn["target_indices"]
    )
    if values.shape != (len(batch["path"]),) or not torch.isfinite(values).all():
        raise ValueError("Invalid SSL loss")
    return values, drawn


def tensor_audit(state, initial, excluded=("pos_embed",)):
    report = []
    for name, value in state.items():
        before = initial[name].detach().to(value.device)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite tensor: " + name)
        changed = int((value != before).sum())
        fixed = name in excluded
        if fixed and changed:
            raise ValueError("Fixed position changed: " + name)
        report.append(
            {
                "name": name,
                "numel": value.numel(),
                "changed_values": changed,
                "fixed": fixed,
            }
        )
    ordinary = [r for r in report if not r["fixed"]]
    return {
        "ordinary_tensors": len(ordinary),
        "ordinary_tensors_changed": sum(r["changed_values"] > 0 for r in ordinary),
        "every_ordinary_tensor_changed": all(r["changed_values"] > 0 for r in ordinary),
        "all_finite": True,
        "fixed_positions_unchanged": True,
        "tensors": report,
    }


def train_ssl(config, roles, output, model, callback, status):
    """Five complete epochs; downstream quality, never latent loss, selects weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = config["training"]
    device = next(model.parameters()).device
    _seed(config["seed"])
    masks = masker(config)
    opt = torch.optim.AdamW(
        model.optimizer_groups(
            settings["encoder_lr"], settings["predictor_lr"], settings["weight_decay"]
        )
    )
    per_epoch = math.ceil(
        math.ceil(len(roles["fitting"]) / settings["batch_size"])
        / settings["gradient_accumulation"]
    )
    total = settings["epochs"] * per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _schedule(s, total, per_epoch)
    )
    counts = Counter(r["group"] for r in roles["fitting"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    history = []
    updates = 0
    best = -1
    initial = {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        iterator = iter(
            loader(roles["fitting"], config, epoch, training=True, ssl=True)
        )
        seen = 0
        numerator = denominator = 0.0
        mask_stats = []
        start = time.monotonic()
        while True:
            window = []
            for _ in range(settings["gradient_accumulation"]):
                try:
                    window.append(next(iterator))
                except StopIteration:
                    break
            if not window:
                break
            opt.zero_grad(set_to_none=True)
            n = sum(len(b["path"]) for b in window)
            for batch in window:
                values, drawn = ssl_loss(model, masks, batch, epoch, 0, device)
                weights = values.new_tensor(
                    [
                        len(roles["fitting"]) / (len(counts) * counts[g])
                        for g in batch["group"]
                    ]
                )
                weighted = (values * weights).sum()
                (weighted / n).backward()
                numerator += float(weighted.detach())
                denominator += float(weights.sum())
                seen += len(values)
                mask_stats.append(
                    {
                        k: v
                        for k, v in drawn.items()
                        if not isinstance(v, torch.Tensor) and k != "target_indices"
                    }
                )
            if any(
                p.grad is None or not torch.isfinite(p.grad).all() for p in trainable
            ):
                raise ValueError("Missing/nonfinite SSL gradient")
            norm = nn.utils.clip_grad_norm_(trainable, settings["gradient_clip"])
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient norm")
            opt.step()
            momentum = settings["ema_start"] + (
                settings["ema_end"] - settings["ema_start"]
            ) * updates / max(1, total - 1)
            model.update_target(momentum)
            updates += 1
            scheduler.step()
            if updates % 50 == 0 or seen == len(roles["fitting"]):
                status(
                    "ssl_training",
                    arm=config["arm"],
                    epoch=epoch,
                    seen=seen,
                    updates=updates,
                    loss=numerator / denominator,
                )
        if seen != len(roles["fitting"]) or updates != epoch * per_epoch:
            raise ValueError("Incomplete SSL epoch")
        # Latent loss is a diagnostic only, with fixed native cropped views.
        model.eval()
        val_num = val_den = 0.0
        val_seen = 0
        val_masks = masker(config)
        val_counts = Counter(r["group"] for r in roles["selection"])
        with torch.no_grad():
            for batch in loader(roles["selection"], config, 0, view=1, ssl=True):
                values, _ = ssl_loss(model, val_masks, batch, 0, 1, device)
                weights = values.new_tensor([1 / val_counts[g] for g in batch["group"]])
                val_num += float((values * weights).sum())
                val_den += float(weights.sum())
                val_seen += len(values)
        if val_seen != len(roles["selection"]):
            raise ValueError("Incomplete latent validation")
        state = rng_state()
        quality = callback(model.target_encoder, epoch, updates)
        restore_rng(state)
        row = {
            "epoch": epoch,
            "updates": updates,
            "fitting_frames": seen,
            "fitting_loss": numerator / denominator,
            "latent_validation": val_num / val_den,
            "quality_selection": quality,
            "seconds": time.monotonic() - start,
        }
        history.append(row)
        payload = {
            "identity": digest(config),
            "config": config,
            "model": model.state_dict(),
            "completed_epochs": epoch,
            "optimizer_updates": updates,
            "history": history,
            "selection_metric": "downstream validation BA",
            "resumable": False,
        }
        if quality["selection_BA"] > best:
            best = quality["selection_BA"]
            atomic_torch(output / "best_weights.pt", payload)
        atomic_torch(output / "last_weights.pt", payload)
        atomic_json(output / "history.json", history)
        atomic_json(output / f"masks_epoch{epoch}.json", mask_stats)
    audit = {}
    current = model.state_dict()
    for role in ("context_encoder", "predictor", "target_encoder"):
        prefix = role + "."
        a = {k[len(prefix) :]: v for k, v in current.items() if k.startswith(prefix)}
        b = {k[len(prefix) :]: v for k, v in initial.items() if k.startswith(prefix)}
        if role == "target_encoder":
            # Existing official role snapshots can carry different fixed tables;
            # EMA deliberately copies the unchanged context sinusoidal table.
            b["pos_embed"] = initial["context_encoder.pos_embed"]
        audit[role] = tensor_audit(a, b, ("pos_embed", "predictor_pos_embed"))
        if not audit[role]["every_ordinary_tensor_changed"]:
            raise ValueError("Unchanged ordinary SSL tensor")
    steps = sorted({int(v["step"]) for v in opt.state.values()})
    if len(opt.state) != len(trainable) or steps != [updates]:
        raise ValueError("SSL optimizer coverage")
    atomic_json(
        output / "Weight_Audit.json",
        {
            "passed": True,
            "roles": audit,
            "optimizer_states": len(opt.state),
            "steps": steps,
        },
    )
    atomic_json(
        output / "complete.json",
        {
            "passed": True,
            "epochs": settings["epochs"],
            "updates": updates,
            "expected_updates": total,
            "quality_selection": True,
            "assessment_accessed": False,
            "checkpoint_scope": "full ctx/teacher/predictor weights; optimizer not retained",
        },
    )
    del initial, opt, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return history


def masker(config):
    if config["ssl_mask_backend"] != "native":
        raise ValueError("This branch contains only the selected native-mask recipe")
    if config["training"]["warmup_epochs"] != 1:
        raise ValueError("The recorded recipe uses one warmup epoch")
    return NativeMasker(config["seed"])


def train(config_path, output, device="cuda"):
    config = load_config(config_path)
    roles = load_roles(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Training requires a new run directory")
    configure_precision()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Fail before allocating a large model if fitting/validation images are absent.
    from .data import ResearchImages
    from .data import preprocessing

    for role in ("fitting", "selection"):
        ResearchImages(roles[role], config["data_root"], preprocessing(config))
    output.mkdir(parents=True)
    atomic_json(output / "config.json", config)
    best = None

    def status(stage, **fields):
        atomic_json(output / "status.json", {"stage": stage, **fields})
        print(json.dumps({"stage": stage, **fields}), flush=True)

    def readout(encoder, epoch, updates):
        nonlocal best
        destination = output / "readouts" / f"epoch{epoch}"
        destination.mkdir(parents=True)
        features = {
            role: extract(encoder, roles[role], config, device)
            for role in ("fitting", "selection")
        }
        probe = fit_probe(
            features["fitting"],
            features["selection"],
            roles["fitting"],
            roles["selection"],
            destination / "mean",
            f"native_epoch{epoch}",
            "mean",
            device,
            recipe=config["classifier"],
        )
        score = probe["score"]
        if best is None or score["selection_BA"] > best["selection_BA"]:
            best = {
                "candidate": f"native_epoch{epoch}",
                "adaptation_epoch": epoch,
                "adaptation_updates": updates,
                "classifier_epoch": probe["epoch"],
                **score,
            }
            atomic_torch(
                output / "encoder.pt",
                {
                    "identity": digest(best),
                    "encoder": encoder.state_dict(),
                    "config": config,
                    "selection": best,
                },
            )
            shutil.copyfile(destination / "mean/best.pt", output / "classifier.pt")
            shutil.copyfile(destination / "mean/last.pt", output / "classifier_last.pt")
            atomic_json(output / "selection.json", best)
        del features, probe
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return score

    with exclusive_run(output):
        model = build_model(config, device)
        if config.get("weld_adaptation", True):
            train_ssl(config, roles, output / "adaptation", model, readout, status)
        else:
            # Original baseline: frozen released target encoder, classifier only.
            model.requires_grad_(False).eval()
            readout(model.target_encoder, 0, 0)
        del model
        gc.collect()
        seal = {
            "format": "ijepa-vit-h14-selection-seal-v1",
            "selected": best,
            "threshold": 0.5,
            "config_sha256": sha256(output / "config.json"),
            "manifest_sha256": config["manifest_sha256"],
            "role_sha256": {role: digest(rows) for role, rows in roles.items()},
            "checkpoint_sha256": {
                "encoder": sha256(output / "encoder.pt"),
                "classifier": sha256(output / "classifier.pt"),
            },
            "final_classifier_sha256": sha256(output / "classifier_last.pt"),
            "classifier_recipe": config["classifier"],
            "assessment_pixels_accessed": False,
        }
        atomic_json(output / "seal.json", seal)
        status("training_complete", selected=best, assessment_pixels_accessed=False)
    return best
