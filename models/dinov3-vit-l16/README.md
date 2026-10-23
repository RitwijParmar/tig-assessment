# DINOv3 ViT-L/16

**Implementation ready; matched training/assessment results are not yet available.** Uses exactly the final adapted I-JEPA fitting/validation/assessment frame memberships: **12 recordings / 10,338 frames**, **6 / 6,454**, **9 / 6,763**. No training-plus-validation refit or random frame split.

## Architecture and weight updates

Released web-pretrained ViT-L/16: 24 blocks, width 1024, patch size 16. Concatenate the final normalized CLS token with the mean of normalized patch tokens, excluding register tokens, for 2048 features.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The encoder remains fixed and only the linear classifier updates, following the declared frozen-feature transfer approach.

## Source and pretrained weights

- Upstream: https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751
- Released checkpoint: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51
- Checkpoint SHA-256: `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- [Exact model/readout config](config.json); [shared implementation and runnable commands](../../tig_benchmark/README.md).

The runtime uses the pinned upstream encoder rather than a rewritten architecture. Sources are verified against Git blob hashes and checkpoint bytes against SHA-256 before use. Upstream source and weight licensing terms apply; weights are obtained separately and are not redistributed here.

## Data, recipe and evaluation

Use the [shared data preparation](../../tig_benchmark/README.md#setup-and-exact-data-preparation), [classifier recipe and evaluation commands](../../tig_benchmark/README.md#fit-seal-and-assess). Raw paths, annotations, labels and checksums must match the [final I-JEPA split fingerprints](../ijepa-vit-h14/config.json). Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`.

The same existing classifier-fitting function, fitting-only normalization, balanced BCE, optimizer, patience, selection metric and threshold are shared with I-JEPA. Model-native preprocessing and its full-frame geometry control are declared separately. Assessment is excluded from fitting, normalization and checkpoint choice, and requires a sealed verified run. Previously inspected recordings provide retrospective assessment rather than a new blind test.

## Status

Source adapters, training, checkpoint verification, assessment and prediction commands are implemented. Saved old MobileNet/DINO results from different folds are not substituted for new matched results. See [verification details](../../tig_benchmark/README.md#verification-and-result-status).

The checkpoint is the official Hugging Face safetensors export. A checked format conversion restores Meta’s state names and concatenated QKV tensors, reshapes the mask token’s singleton axis, and retains deterministic RoPE/key-bias-mask buffers from the upstream model. No learned values are changed or unexplained keys dropped. The official readout helper is executed unchanged without importing the distributed ImageNet training CLI.
