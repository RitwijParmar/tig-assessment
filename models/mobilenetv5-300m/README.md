# MobileNetV5-300M

**Implementation ready; matched training/assessment results are not yet available.** Uses exactly the final adapted I-JEPA fitting/validation/assessment frame memberships: **12 recordings / 10,338 frames**, **6 / 6,454**, **9 / 6,763**. No training-plus-validation refit or random frame split.

## Architecture and weight updates

Released Gemma 3n vision encoder, including its native multiscale fusion adapter and global pooling. Native 768px evaluation yields a 2048-dimensional image vector.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The primary encoder remains fixed; only the linear classifier updates. A separate optional supervised arm updates the encoder and classifier, retaining the frozen baseline as epoch 0.

## Source and pretrained weights

- Upstream: https://github.com/huggingface/pytorch-image-models/tree/af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c
- Released checkpoint: https://huggingface.co/timm/mobilenetv5_300m.gemma3n/tree/f3a689df9847019f95abea17423a680b3d4512ac
- Checkpoint SHA-256: `affbbd57927c4138b783564d98603ef54530eb02ddf25cbabf79a4b2ce0fb321`.
- [Exact model/readout config](config.json); [shared implementation and runnable commands](../../tig_benchmark/README.md).

The runtime uses the pinned upstream encoder rather than a rewritten architecture. Sources are verified against Git blob hashes and checkpoint bytes against SHA-256 before use. Upstream source and weight licensing terms apply; weights are obtained separately and are not redistributed here.

## Data, recipe and evaluation

Use the [shared data preparation](../../tig_benchmark/README.md#setup-and-exact-data-preparation), [classifier recipe and evaluation commands](../../tig_benchmark/README.md#fit-seal-and-assess). Raw paths, annotations, labels and checksums must match the [final I-JEPA split fingerprints](../ijepa-vit-h14/config.json). Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`.

The same existing classifier-fitting function, fitting-only normalization, balanced BCE, optimizer, patience, selection metric and threshold are shared with I-JEPA. Model-native preprocessing and its full-frame geometry control are declared separately. Assessment is excluded from fitting, normalization and checkpoint choice, and requires a sealed verified run. Previously inspected recordings provide retrospective assessment rather than a new blind test.

## Status

Source adapters, training, checkpoint verification, assessment and prediction commands are implemented. Saved old MobileNet/DINO results from different folds are not substituted for new matched results. See [verification details](../../tig_benchmark/README.md#verification-and-result-status).
