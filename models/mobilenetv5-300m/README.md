# MobileNetV5-300M + linear classifier

## Status

Implementation ready; matched weld training and assessment have not run. [model.json](model.json) gives the common package metadata. [config.json](config.json) pins the released model, preprocessing/readout and fitting recipe. [results.json](results.json) records unavailable metrics explicitly.

## Architecture

Released Gemma 3n vision encoder, including its native multiscale fusion adapter and global pooling. Native 768px evaluation yields a 2048-dimensional image vector.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The primary encoder remains fixed; only the linear classifier updates. A separate optional supervised arm updates the encoder and classifier, retaining the frozen baseline as epoch 0.

## Data

Exactly the final adapted I-JEPA recording split: fitting **12 recordings / 10,338 frames**, validation **6 / 6,454**, assessment **9 / 6,763**. [model.json](model.json) includes class counts and exact normalized-row hashes. All recording segments remain together; no random frame split or fitting-plus-validation refit is used.

Dataset: [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`. Assessment recordings were inspected in earlier research, so future scores remain retrospective held-out results.

## Training and selection

Use the [shared data preparation](../../tig_benchmark/README.md#setup-and-exact-data-preparation), [classifier recipe and evaluation commands](../../tig_benchmark/README.md#fit-seal-and-assess). Raw paths, annotations, labels and checksums must match the [final I-JEPA split fingerprints](../ijepa-vit-h14/config.json). Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`.

The same existing classifier-fitting function, fitting-only normalization, balanced BCE, optimizer, patience, selection metric and threshold are shared with I-JEPA. Model-native preprocessing and its full-frame geometry control are declared separately. Assessment is excluded from fitting, normalization and checkpoint choice, and requires a sealed verified run. Previously inspected recordings provide retrospective assessment rather than a new blind test.

## Results

No matched weld-classification metrics are available. [results.json](results.json) has null validation and assessment entries. Existing historical scores from other recording splits are not substituted. Real pretrained CPU forward checks passed; these are implementation checks, not weld-classification results.

## Usage

Run every command from the repository root. Follow the shared [setup and data preparation](../../tig_benchmark/README.md#setup-and-exact-data-preparation), [asset registration](../../tig_benchmark/README.md#assets), and [training, verification, assessment and prediction commands](../../tig_benchmark/README.md#fit-seal-and-assess). Install [requirements-benchmark.txt](../../requirements-benchmark.txt); run `python -m pytest -q tests`.

## Weights

The released pretrained encoder is saved locally under [weights/](weights/). [weights/manifest.json](weights/manifest.json) records its relative filename, size, SHA-256 and the unavailable weld classifier. Future matched runs retain selected and final-epoch artifacts separately. Checkpoint bytes remain outside Git.

## Implementation and licenses

- Upstream: https://github.com/huggingface/pytorch-image-models/tree/af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c
- Released checkpoint: https://huggingface.co/timm/mobilenetv5_300m.gemma3n/tree/f3a689df9847019f95abea17423a680b3d4512ac
- Checkpoint SHA-256: `affbbd57927c4138b783564d98603ef54530eb02ddf25cbabf79a4b2ce0fb321`.
- [Exact model/readout config](config.json); [shared implementation and runnable commands](../../tig_benchmark/README.md).

The runtime uses the pinned upstream encoder rather than a rewritten architecture. Sources are verified against Git blob hashes and checkpoint bytes against SHA-256 before use. Upstream source and weight licensing terms apply; weights are obtained separately and are not redistributed here.


