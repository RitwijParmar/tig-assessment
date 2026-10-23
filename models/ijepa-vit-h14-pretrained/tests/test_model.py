"""Real tiny I-JEPA contracts; synthetic wiring evidence, not weld accuracy."""

from __future__ import annotations

import copy
from functools import partial

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ijepa.vendor.ijepa.vision_transformer import (
    VisionTransformer,
    VisionTransformerPredictor,
)
from ijepa.model import IJEPAAdapter


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(seed=71):
    torch.manual_seed(seed)
    norm = partial(nn.LayerNorm, eps=1e-6)
    context = VisionTransformer(
        img_size=[32],
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=norm,
    )
    target = copy.deepcopy(context)
    # The original checkpoint's context and EMA-target weights need not coincide.
    with torch.no_grad():
        target.patch_embed.proj.weight.add_(0.013)
    predictor = VisionTransformerPredictor(
        num_patches=16,
        embed_dim=32,
        predictor_embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=norm,
    )
    return IJEPAAdapter(
        context,
        target,
        predictor,
        {"image_size": 32, "patch_size": 8},
        {"source_commit": "tiny-original-ijepa-fixture"},
    )


def batch():
    generator = torch.Generator().manual_seed(19)
    images = torch.randn(2, 3, 32, 32, generator=generator)
    context = torch.tensor([[0, 1, 4, 5, 8, 9], [2, 3, 6, 7, 10, 11]])
    targets = [
        torch.tensor([[2, 3], [0, 1]]),
        torch.tensor([[6, 7], [4, 5]]),
        torch.tensor([[10, 11], [8, 9]]),
        torch.tensor([[14, 15], [12, 13]]),
    ]
    return images, context, targets


def test_ssl_trains_original_context_and_predictor_with_no_classifier_or_lora():
    model = tiny_model().train()
    images, context, targets = batch()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    losses = model(images, context, targets)
    assert losses.shape == (2,) and torch.isfinite(losses).all()
    assert (losses > 0).all() and losses.requires_grad
    losses.mean().backward()

    for role in ("context_encoder", "predictor"):
        component = getattr(model, role)
        for name, parameter in component.named_parameters():
            fixed = name in {"pos_embed", "predictor_pos_embed"}
            assert parameter.requires_grad is not fixed, (role, name)
            if fixed:
                assert parameter.grad is None
            else:
                assert parameter.grad is not None, (role, name)
                assert torch.isfinite(parameter.grad).all(), (role, name)
        blocks = (
            component.blocks
            if role == "context_encoder"
            else component.predictor_blocks
        )
        assert all(block.attn.qkv.weight.grad.abs().sum() > 0 for block in blocks)
    assert model.context_encoder.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.predictor.mask_token.grad.abs().sum() > 0
    assert all(
        not p.requires_grad and p.grad is None
        for p in model.target_encoder.parameters()
    )
    assert not any(
        "lora" in name or "head" in name for name, _ in model.named_parameters()
    )

    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 2e-3, 0.01))
    optimizer.step()
    for role in ("context_encoder", "predictor"):
        assert any(
            not torch.equal(parameter, before[role + "." + name])
            for name, parameter in getattr(model, role).named_parameters()
            if parameter.requires_grad
        )
    for name, parameter in model.named_parameters():
        if name.startswith("target_encoder.") or name.endswith("pos_embed"):
            assert torch.equal(parameter, before[name]), name


def test_first_context_attention_sees_only_visible_patches_teacher_sees_full_image():
    model = tiny_model().train()
    images, context, targets = batch()
    seen = {}

    def capture(role):
        def hook(_module, args):
            seen[role] = (args[0].detach().clone(), torch.is_grad_enabled())

        return hook

    handles = [
        model.context_encoder.blocks[0].register_forward_pre_hook(capture("context")),
        model.target_encoder.blocks[0].register_forward_pre_hook(capture("target")),
    ]
    try:
        model(images, context, targets)
    finally:
        for handle in handles:
            handle.remove()
    assert seen["context"][0].shape == (2, 6, 32)
    assert seen["target"][0].shape == (2, 16, 32)
    assert seen["context"][1] is True and seen["target"][1] is False
    with torch.no_grad():
        all_context = (
            model.context_encoder.patch_embed(images) + model.context_encoder.pos_embed
        )
        expected = all_context.gather(1, context[..., None].expand(-1, -1, 32))
    torch.testing.assert_close(seen["context"][0], expected, rtol=0, atol=0)


def test_perturbing_hidden_pixels_cannot_change_predictor_but_changes_teacher_loss():
    model = tiny_model().eval()
    images, context, targets = batch()
    altered = images.clone()
    for sample in range(len(images)):
        for patch_index in set(range(16)) - set(context[sample].tolist()):
            row, column = divmod(patch_index, 4)
            altered[
                sample, :, row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8
            ] += 20
    predictions = []
    handle = model.predictor.register_forward_hook(
        lambda _module, _args, output: predictions.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            original_loss = model(images, context, targets)
            changed_loss = model(altered, context, targets)
    finally:
        handle.remove()
    assert len(predictions) == 2
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
    assert not torch.allclose(original_loss, changed_loss)


def test_multiblock_batch_order_matches_independent_single_block_oracle():
    model = tiny_model().eval()
    images, context, targets = batch()
    with torch.no_grad():
        actual = model(images, context, targets)
        # Independent block-by-block calls catch accidental image/block reshape swaps.
        full_teacher = F.layer_norm(model.target_encoder(images).float(), (32,))
        visible = model.context_encoder(images, masks=context)
        per_block = []
        for target in targets:
            prediction = model.predictor(visible, context, [target])
            expected = full_teacher.gather(1, target[..., None].expand(-1, -1, 32))
            per_block.append(
                F.smooth_l1_loss(
                    prediction.float(), expected, reduction="none", beta=1
                ).mean(dim=(1, 2))
            )
        oracle = torch.stack(per_block).mean(dim=0)
        reversed_blocks = model(images, context, list(reversed(targets)))
        permutation = torch.tensor([1, 0])
        permuted_batch = model(
            images[permutation],
            context[permutation],
            [target[permutation] for target in targets],
        )
    torch.testing.assert_close(actual, oracle, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(reversed_blocks, actual, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        permuted_batch, actual[permutation], rtol=1e-5, atol=1e-7
    )


def test_target_ema_uses_post_optimizer_context_without_aliasing_or_implicit_update():
    model = tiny_model().train()
    images, context, targets = batch()
    context_state = dict(model.context_encoder.named_parameters())
    target_state = dict(model.target_encoder.named_parameters())
    assert context_state.keys() == target_state.keys()
    assert all(
        context_state[name].data_ptr() != value.data_ptr()
        for name, value in target_state.items()
    )
    teacher_before = {name: p.detach().clone() for name, p in target_state.items()}
    optimizer = torch.optim.SGD(model.optimizer_groups(0.03, 0.03, 0))
    model(images, context, targets).mean().backward()
    optimizer.step()
    assert all(torch.equal(p, teacher_before[name]) for name, p in target_state.items())
    context_after = {name: p.detach().clone() for name, p in context_state.items()}
    model.update_target(0.8)
    for name, value in target_state.items():
        expected = teacher_before[name] * 0.8 + context_after[name] * 0.2
        torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)
        assert not value.requires_grad and value.grad is None
    assert all(torch.equal(p, context_after[name]) for name, p in context_state.items())
    model.update_target(0.0)
    assert all(
        torch.equal(value, context_state[name]) for name, value in target_state.items()
    )
    before_noop = {name: p.detach().clone() for name, p in target_state.items()}
    model.update_target(1.0)
    assert all(torch.equal(p, before_noop[name]) for name, p in target_state.items())
    assert model.target_encoder.training is False


def test_optimizer_exact_scope_decay_and_lr_and_parameter_accounting():
    model = tiny_model()
    groups = model.optimizer_groups(1e-4, 3e-4, 0.05)
    actual = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(actual) == len(set(actual)) and set(actual) == expected
    names = {id(p): name for name, p in model.named_parameters()}
    for group in groups:
        for parameter in group["params"]:
            name = names[id(parameter)]
            assert group["lr"] == (
                1e-4 if name.startswith("context_encoder.") else 3e-4
            )
            assert group["weight_decay"] == (
                0.0 if parameter.ndim <= 1 or name.endswith(".bias") else 0.05
            )
    counts = model.parameter_counts()
    assert counts["trainable"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert counts["total"] == sum(p.numel() for p in model.parameters())


def test_full_state_checkpoint_roundtrip_preserves_both_encoders_and_predictor(
    tmp_path,
):
    model = tiny_model().train()
    images, context, targets = batch()
    optimizer = torch.optim.AdamW(model.optimizer_groups(0.001, 0.002, 0))
    model(images, context, targets).mean().backward()
    optimizer.step()
    model.update_target(0.9)
    model.eval()
    with torch.no_grad():
        expected = model(images, context, targets)
    checkpoint = tmp_path / "jepa.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_spec": model.model_spec,
            "provenance": model.provenance,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = tiny_model(seed=72).eval()
    restored.load_state_dict(loaded["model"], strict=True)
    with torch.no_grad():
        actual = restored(images, context, targets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded["model_spec"] == model.model_spec
    assert loaded["provenance"] == model.provenance
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name


@pytest.mark.parametrize(
    "case",
    [
        "overlap",
        "duplicate_context",
        "duplicate_target",
        "range",
        "float_context",
        "float_target",
        "target_count",
        "batch",
    ],
)
def test_reject_invalid_masks_before_learning(case):
    model = tiny_model()
    images, context, targets = batch()
    if case == "overlap":
        targets[0][0, 0] = context[0, 0]
    elif case == "duplicate_context":
        context[0, 1] = context[0, 0]
    elif case == "duplicate_target":
        targets[0][0, 1] = targets[0][0, 0]
    elif case == "range":
        targets[0][0, 0] = 16
    elif case == "float_context":
        context = context.float()
    elif case == "float_target":
        targets[0] = targets[0].float()
    elif case == "target_count":
        targets.pop()
    elif case == "batch":
        context = context[:1]
    with pytest.raises((ValueError, TypeError)):
        model(images, context, targets)


def test_target_blocks_may_overlap_each_other():
    model = tiny_model().eval()
    images, context, targets = batch()
    targets[1] = targets[0].clone()
    assert torch.isfinite(model(images, context, targets)).all()


def test_checkpoint_storage_alias_is_broken_before_training():
    reference = tiny_model()
    context = copy.deepcopy(reference.context_encoder)
    target = copy.deepcopy(context)
    target.load_state_dict(context.state_dict(), assign=True)
    assert (
        target.patch_embed.proj.weight.data_ptr()
        == context.patch_embed.proj.weight.data_ptr()
    )
    model = IJEPAAdapter(
        context,
        target,
        copy.deepcopy(reference.predictor),
        {"image_size": 32, "patch_size": 8},
        {},
    )
    assert (
        model.target_encoder.patch_embed.proj.weight.data_ptr()
        != context.patch_embed.proj.weight.data_ptr()
    )
    teacher_before = model.target_encoder.patch_embed.proj.weight.detach().clone()
    with torch.no_grad():
        context.patch_embed.proj.weight.add_(1.0)
    assert torch.equal(model.target_encoder.patch_embed.proj.weight, teacher_before)
    with pytest.raises(ValueError, match="distinct"):
        IJEPAAdapter(
            context,
            context,
            copy.deepcopy(reference.predictor),
            {"image_size": 32, "patch_size": 8},
            {},
        )


def test_ssl_input_gradient_only_reaches_visible_pixels():
    model = tiny_model().train()
    images, context, targets = batch()
    images.requires_grad_(True)
    model(images, context, targets).mean().backward()
    assert images.grad is not None
    for sample in range(len(images)):
        visible = set(context[sample].tolist())
        visible_gradient = 0.0
        for index in range(16):
            row, column = divmod(index, 4)
            patch_grad = images.grad[
                sample, :, row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8
            ]
            if index in visible:
                visible_gradient += float(patch_grad.abs().sum())
            else:
                assert torch.count_nonzero(patch_grad) == 0
        assert visible_gradient > 0


@pytest.mark.parametrize("momentum", [-0.01, 1.01, float("nan"), float("inf")])
def test_reject_invalid_ema_momentum(momentum):
    with pytest.raises((ValueError, TypeError)):
        tiny_model().update_target(momentum)
