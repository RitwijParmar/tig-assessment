"""Two pretrained encoders, masked predictor loss, and EMA target updates."""

from collections.abc import Mapping
import copy, hashlib, json, math, re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.ijepa.vision_transformer import vit_huge, vit_predictor

SOURCE_COMMIT = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
VENDOR = Path(__file__).resolve().parent / "vendor"
FORMAT = "autonoweld-original-ijepa-full-ssl-v1"
TARGET_BLOCKS = 4
_FIXED_POSITIONS = {"context_encoder.pos_embed", "predictor.predictor_pos_embed"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_vendor():
    path = VENDOR / "SOURCE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    if manifest["source_commit"] != SOURCE_COMMIT:
        raise ValueError("Vendored I-JEPA source commit mismatch")
    hashes = {}
    for item in manifest["files"]:
        actual = sha256_file(VENDOR / item["file"])
        if actual != item["local_sha256"]:
            raise ValueError(f"Vendored source hash mismatch: {item['file']}")
        hashes[item["file"]] = actual
    return {
        "source_commit": SOURCE_COMMIT,
        "license": "CC BY-NC 4.0",
        "source_manifest_sha256": sha256_file(path),
        "vendored_file_sha256": hashes,
    }


def _strict_load(module, raw, role):
    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint has no tensor dictionary for {role}")
    cleaned = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Non-tensor checkpoint state in {role}")
        name = key
        while name.startswith("module."):
            name = name[len("module.") :]
        if name in cleaned:
            raise ValueError(f"Colliding normalized {role} key: {name}")
        cleaned[name] = value
    expected = module.state_dict()
    missing = sorted(expected.keys() - cleaned.keys())
    unexpected = sorted(cleaned.keys() - expected.keys())
    wrong_shape = {
        name: {
            "expected": list(expected[name].shape),
            "actual": list(cleaned[name].shape),
        }
        for name in expected.keys() & cleaned.keys()
        if expected[name].shape != cleaned[name].shape
    }
    if missing or unexpected or wrong_shape:
        raise ValueError(
            f"Strict {role} state audit failed: "
            + json.dumps(
                {
                    "missing": missing,
                    "unexpected": unexpected,
                    "shape_errors": wrong_shape,
                }
            )
        )
    module.load_state_dict(cleaned, strict=True, assign=True)
    return {
        "role": role,
        "strict": True,
        "tensor_count": len(cleaned),
        "missing": [],
        "unexpected": [],
        "checkpoint_dtypes": sorted({str(value.dtype) for value in cleaned.values()}),
    }


def _storage_key(value):
    if value.device.type == "meta":
        raise ValueError("I-JEPA requires materialized model tensors")
    return (str(value.device), value.untyped_storage().data_ptr())


def _detach_shared_storage(module, others):
    """Break mmap/checkpoint aliases without copying independent huge tensors."""
    occupied = {
        _storage_key(value)
        for other in others
        for value in (*other.parameters(), *other.buffers())
        if value.numel()
    }
    copied = []
    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        if value.numel() and _storage_key(value) in occupied:
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            detached = value.detach().clone()
            if leaf in parent._parameters:
                parent._parameters[leaf] = nn.Parameter(
                    detached, requires_grad=value.requires_grad
                )
            else:
                parent._buffers[leaf] = detached
            copied.append(name)
    return copied


class IJEPAAdapter(nn.Module):
    """One context and four target blocks, with per-image SSL losses [B].

    Direct construction accepts small native ViT fixtures for verification.
    Production build_model always constructs the exact original H/14 triple.
    Target blocks may overlap each other, as in the original multi-block method;
    every block must be disjoint from the visible context.
    """

    def __init__(self, context_encoder, target_encoder, predictor, config, provenance):
        super().__init__()
        if not isinstance(config, Mapping) or not isinstance(provenance, Mapping):
            raise TypeError("I-JEPA config and provenance must be mappings")
        if len({id(context_encoder), id(target_encoder), id(predictor)}) != 3:
            raise ValueError("Context, target and predictor must be distinct modules")
        runtime = config.get("runtime", {})
        if not isinstance(runtime, Mapping):
            raise TypeError("runtime must be a mapping")
        if runtime.get("attention_backend", "manual") != "manual" or runtime.get(
            "gradient_checkpointing", False
        ):
            raise ValueError(
                "This verified original I-JEPA implementation supports manual attention without gradient checkpointing"
            )
        self.image_size = config.get("image_size", 224)
        self.patch_size = config.get("patch_size", 14)
        if (
            not isinstance(self.image_size, int)
            or isinstance(self.image_size, bool)
            or not isinstance(self.patch_size, int)
            or isinstance(self.patch_size, bool)
            or self.image_size <= 0
            or self.patch_size <= 0
            or self.image_size % self.patch_size
        ):
            raise ValueError(
                "image_size must be a positive integer multiple of patch_size"
            )
        self.token_count = (self.image_size // self.patch_size) ** 2
        self.feature_dim = context_encoder.embed_dim
        if target_encoder.embed_dim != self.feature_dim:
            raise ValueError(
                "Matched context and target encoders require equal feature dimensions"
            )
        if (
            tuple(context_encoder.pos_embed.shape)
            != (1, self.token_count, self.feature_dim)
            or tuple(target_encoder.pos_embed.shape)
            != tuple(context_encoder.pos_embed.shape)
            or predictor.predictor_pos_embed.shape[1] != self.token_count
            or predictor.predictor_embed.in_features != self.feature_dim
            or predictor.predictor_proj.out_features != self.feature_dim
        ):
            raise ValueError(
                "Encoder/predictor patch and feature dimensions do not match"
            )
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.native_mask_fallback = config.get("ssl_mask_backend") == "native"
        if any("lora_" in name.lower() for name, _ in self.named_parameters()):
            raise ValueError("Original I-JEPA does not accept LoRA parameters")
        # Separate roles must stay independent even if a serialized checkpoint
        # happened to share storage. No target tensor may change at Ex's step.
        predictor_copies = _detach_shared_storage(
            self.predictor, [self.context_encoder]
        )
        target_copies = _detach_shared_storage(
            self.target_encoder, [self.context_encoder, self.predictor]
        )
        self.provenance = copy.deepcopy(dict(provenance))
        self.provenance["storage_alias_copies"] = {
            "predictor": predictor_copies,
            "target_encoder": target_copies,
        }
        self.context_encoder.requires_grad_(True)
        self.predictor.requires_grad_(True)
        self.context_encoder.pos_embed.requires_grad_(False)
        self.predictor.predictor_pos_embed.requires_grad_(False)
        self.target_encoder.requires_grad_(False)
        for parameter in self.target_encoder.parameters():
            parameter.grad = None
        self.model_spec = {
            "format": FORMAT,
            "architecture": "original image I-JEPA",
            "source_commit": SOURCE_COMMIT,
            "context_role": "encoder",
            "target_role": "target_encoder",
            "predictor_role": "predictor",
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "patch_tokens": self.token_count,
            "feature_dim": self.feature_dim,
            "context_depth": len(context_encoder.blocks),
            "predictor_depth": len(predictor.predictor_blocks),
            "predictor_embed_dim": predictor.predictor_embed.out_features,
            "context_masking": "before self-attention",
            "target_blocks": TARGET_BLOCKS,
            "target_block_order": "block-major [block0 batch, block1 batch, ...]",
            "objective": "SmoothL1_beta1_feature_normalized_full_image_teacher_targets",
            "loss_reduction": "equal mean over four blocks, tokens and dimensions per image",
            "trainable_rule": "all ordinary context encoder and predictor parameters; fixed sinusoidal positions excluded",
            "target_update": "stop-gradient EMA after successful optimizer update",
            "attention_backend": "manual",
            "gradient_checkpointing": False,
        }
        self.last_statistics = {}
        self.parameter_counts()
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def _validate_inputs(self, images, context_masks, target_masks):
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or not len(images)
            or tuple(images.shape[1:]) != (3, self.image_size, self.image_size)
            or not images.is_floating_point()
            or not bool(torch.isfinite(images).all())
        ):
            raise ValueError(
                "Expected finite normalized RGB images [B,3,image_size,image_size]"
            )
        if not isinstance(target_masks, list) or len(target_masks) != TARGET_BLOCKS:
            raise ValueError("Exactly four target mask tensors are required")
        masks = [context_masks, *target_masks]
        for mask in masks:
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.long
                or mask.ndim != 2
                or mask.shape[0] != len(images)
                or not mask.shape[1]
            ):
                raise ValueError(
                    "Every context/target mask must be a nonempty LongTensor [B,K]"
                )
            if bool((mask < 0).any()) or bool((mask >= self.token_count).any()):
                raise ValueError("Mask index outside the patch grid")
            ordered = mask.sort(dim=1).values
            if ordered.shape[1] > 1 and bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                raise ValueError(
                    "Duplicate patch indices inside a context or target block"
                )
        if len({mask.shape[1] for mask in target_masks}) != 1:
            raise ValueError(
                "Official multi-block predictor requires equal target block lengths"
            )
        context_masks = context_masks.to(images.device)
        target_masks = [mask.to(images.device) for mask in target_masks]
        for target in target_masks:
            if (
                bool((context_masks[:, :, None] == target[:, None, :]).any())
                and not self.native_mask_fallback
            ):
                raise ValueError(
                    "Visible context and all target blocks must be disjoint"
                )
        return context_masks, target_masks

    def forward(self, images, context_masks, target_masks):
        context_masks, target_masks = self._validate_inputs(
            images, context_masks, target_masks
        )
        batch = len(images)
        # no_grad is intentional: target encoder is only an EMA teacher. It
        # receives the complete image, while no hidden token reaches Ex.
        self.target_encoder.eval()
        with torch.no_grad():
            full = self.target_encoder(images)
            if tuple(full.shape) != (batch, self.token_count, self.feature_dim):
                raise ValueError("Unexpected full-image target encoder output")
            normalized = F.layer_norm(full.float(), (self.feature_dim,))
            expected = torch.cat(
                [
                    normalized.gather(
                        1, mask[..., None].expand(-1, -1, self.feature_dim)
                    )
                    for mask in target_masks
                ],
                dim=0,
            )
        visible = self.context_encoder(images, masks=context_masks)
        if tuple(visible.shape) != (batch, context_masks.shape[1], self.feature_dim):
            raise ValueError("Unexpected masked context encoder output")
        # The native predictor repeats the complete batch once per target block.
        # cat/gather above has that same block-major order, not image-major order.
        predicted = self.predictor(visible, context_masks, target_masks)
        if tuple(predicted.shape) != tuple(expected.shape):
            raise ValueError("Predictor output and ordered teacher targets differ")
        elementwise = F.smooth_l1_loss(
            predicted.float(), expected.detach(), beta=1.0, reduction="none"
        )
        losses = elementwise.reshape(
            TARGET_BLOCKS, batch, target_masks[0].shape[1], self.feature_dim
        ).mean(dim=(0, 2, 3))
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError("Nonfinite original I-JEPA SSL loss")
        with torch.no_grad():
            self.last_statistics = {
                "loss_mean": float(losses.mean()),
                "prediction_feature_variance": float(
                    predicted.detach().float().var(dim=(0, 1), unbiased=False).mean()
                ),
                "target_feature_variance": float(
                    expected.float().var(dim=(0, 1), unbiased=False).mean()
                ),
                "context_tokens_per_image": context_masks.shape[1],
                "target_tokens_per_block": target_masks[0].shape[1],
                "target_blocks": TARGET_BLOCKS,
            }
        return losses

    @torch.no_grad()
    def update_target(self, momentum):
        """EMA ordinary parameters and copy fixed positions/buffers after a step."""
        if (
            isinstance(momentum, bool)
            or not isinstance(momentum, (float, int))
            or not math.isfinite(momentum)
            or not 0 <= momentum <= 1
        ):
            raise ValueError("EMA momentum must be finite and in [0,1]")
        online = dict(self.context_encoder.named_parameters())
        target = dict(self.target_encoder.named_parameters())
        online_buffers = dict(self.context_encoder.named_buffers())
        target_buffers = dict(self.target_encoder.named_buffers())
        if (
            online.keys() != target.keys()
            or online_buffers.keys() != target_buffers.keys()
        ):
            raise ValueError("Context and target EMA tensor names differ")
        for sources, destinations in (
            (online, target),
            (online_buffers, target_buffers),
        ):
            for name, value in sources.items():
                other = destinations[name]
                if (
                    value.shape != other.shape
                    or value.dtype != other.dtype
                    or value.device != other.device
                ):
                    raise ValueError(f"Context and target EMA tensor mismatch: {name}")
                if value.numel() and _storage_key(value) == _storage_key(other):
                    raise ValueError(f"Context/target storage alias: {name}")
        for name, parameter in target.items():
            if name == "pos_embed":
                parameter.copy_(
                    online[name]
                )  # Preserve exact fixed sinusoidal positions.
            else:
                parameter.mul_(momentum).add_(
                    online[name].detach(), alpha=1.0 - momentum
                )
            parameter.grad = None
        for name, buffer in target_buffers.items():
            buffer.copy_(online_buffers[name])
        self.target_encoder.eval()

    def parameter_counts(self):
        counts = {
            "context_encoder": 0,
            "predictor": 0,
            "target_encoder": 0,
            "fixed_positional": 0,
            "trainable": 0,
            "total": 0,
        }
        for name, parameter in self.named_parameters():
            is_fixed = name in _FIXED_POSITIONS
            intended = not name.startswith("target_encoder.") and not is_fixed
            if parameter.requires_grad != intended:
                raise ValueError(
                    f"Incorrect original I-JEPA requires_grad state: {name}"
                )
            size = parameter.numel()
            counts[name.split(".", 1)[0]] += size
            counts["total"] += size
            counts["trainable"] += size if intended else 0
            counts["fixed_positional"] += size if is_fixed else 0
        return counts

    def optimizer_groups(self, encoder_lr, predictor_lr, weight_decay):
        """Official AdamW bias/1D decay exclusions, with explicit role LRs."""
        self.parameter_counts()
        for label, value in (
            ("encoder_lr", encoder_lr),
            ("predictor_lr", predictor_lr),
            ("weight_decay", weight_decay),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (label != "weight_decay" and value == 0)
            ):
                raise ValueError(f"Invalid optimizer {label}")
        groups = []
        for role, module, rate in (
            ("context_encoder", self.context_encoder, encoder_lr),
            ("predictor", self.predictor, predictor_lr),
        ):
            for excluded in (False, True):
                parameters = [
                    p
                    for n, p in module.named_parameters()
                    if p.requires_grad and (("bias" in n or p.ndim == 1) == excluded)
                ]
                if parameters:
                    groups.append(
                        {
                            "params": parameters,
                            "lr": rate,
                            "weight_decay": 0.0 if excluded else weight_decay,
                            "WD_exclude": excluded,
                            "name": role + ("_no_decay" if excluded else "_decay"),
                        }
                    )
        observed = [id(p) for group in groups for p in group["params"]]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(
                "Original I-JEPA optimizer allowlist differs from trainable parameters"
            )
        return groups


def build_model(config, device="cpu"):
    """Hash-verify and strictly load the matched pretrained H/14 Ex/Ey/P roles."""
    if not isinstance(config, Mapping):
        raise TypeError("I-JEPA config must be a mapping")
    if config.get("source_commit", SOURCE_COMMIT) != SOURCE_COMMIT:
        raise ValueError("Original I-JEPA source commit mismatch")
    if config.get("image_size", 224) != 224 or config.get("patch_size", 14) != 14:
        raise ValueError(
            "The original matched H/14 checkpoint requires image224/patch14"
        )
    expected = str(config.get("base_checkpoint_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Full base_checkpoint_sha256 required")
    path = Path(config["base_checkpoint_path"]).expanduser().resolve()
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("Original I-JEPA base checkpoint SHA256 mismatch")
    provenance = _verify_vendor()
    with torch.device("meta"):
        context = vit_huge(img_size=[224], patch_size=14)
        target = vit_huge(img_size=[224], patch_size=14)
        predictor = vit_predictor(
            num_patches=256,
            embed_dim=1280,
            predictor_embed_dim=384,
            depth=12,
            num_heads=16,
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Original I-JEPA checkpoint must contain a matched role dictionary"
        )
    audits = {
        role: _strict_load(module, checkpoint.get(role), role)
        for role, module in (
            ("encoder", context),
            ("target_encoder", target),
            ("predictor", predictor),
        )
    }
    provenance.update(
        {
            "base_checkpoint_path": str(path),
            "base_checkpoint_sha256": actual,
            "base_checkpoint_filename": path.name,
            "published_checkpoint_url": CHECKPOINT_URL,
            "same_snapshot_roles": ["encoder", "target_encoder", "predictor"],
            "checkpoint_audits": audits,
            "objective_source": f"https://github.com/facebookresearch/ijepa/blob/{SOURCE_COMMIT}/src/train.py",
        }
    )
    del checkpoint
    return IJEPAAdapter(context, target, predictor, config, provenance).to(
        device=device, dtype=torch.float32
    )
