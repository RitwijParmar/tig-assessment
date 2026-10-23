# DINOv3 ViT-L/16 + linear classifier

## Status

Implementation ready; matched weld training and assessment have not run. [model.json](model.json) gives the common package metadata. [config.json](config.json) pins the released model, preprocessing/readout and fitting recipe. [results.json](results.json) records unavailable metrics explicitly.

## Architecture

Released web-pretrained ViT-L/16: 24 blocks, width 1024, patch size 16. Concatenate the final normalized CLS token with the mean of normalized patch tokens, excluding register tokens, for 2048 features.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The encoder remains fixed and only the linear classifier updates, following the declared frozen-feature transfer approach.

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

- Upstream: https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751
- Released checkpoint: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51
- Checkpoint SHA-256: `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
- [Exact model/readout config](config.json); [shared implementation and runnable commands](../../tig_benchmark/README.md).

The runtime uses the pinned upstream encoder rather than a rewritten architecture. Sources are verified against Git blob hashes and checkpoint bytes against SHA-256 before use. Upstream source and weight licensing terms apply; weights are obtained separately and are not redistributed here.

The checkpoint is the official Hugging Face safetensors export. A checked format conversion restores Meta’s state names and concatenated QKV tensors, reshapes the mask token’s singleton axis, and retains deterministic RoPE/key-bias-mask buffers from the upstream model. No learned values are changed or unexplained keys dropped. The official readout helper is executed unchanged without importing the distributed ImageNet training CLI.
