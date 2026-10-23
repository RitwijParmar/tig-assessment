# MobileNetV5-300M

## Status

Completed. The evaluated checkpoint is in `weights/best/`. [config.json](config.json), [data.json](data.json) and [results.json](results.json) record the architecture, recipe, shared split and measured results.

## Architecture

Released Gemma 3n vision encoder, including its native multiscale fusion adapter and global pooling. Native 768px evaluation yields a 2048-dimensional image vector.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The selected checkpoint updates the complete MobileNetV5 encoder and the linear classifier by gradients during supervised fitting. A frozen-encoder classifier was trained first and supplied the classifier initialization and epoch-0 baseline. Validation selected the final configuration.

## Data

Exactly the final adapted I-JEPA recording split: fitting **12 recordings / 10,338 frames**, validation **6 / 6,454**, assessment **9 / 6,763**. [data.json](data.json) includes class counts and exact normalized-row hashes. All recording segments remain together; no random frame split or fitting-plus-validation refit is used. These canonical recording roles supersede the original archive folder names; a path containing `train` or `test` is not itself a role assignment for this benchmark.

Dataset: [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`. Assessment recordings were inspected in earlier research, so these scores are retrospective held-out results.

## Training and selection

Use the shared data preparation, classifier recipe and evaluation commands. Raw paths, annotations, labels and checksums must match the [split fingerprints](data.json). Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`.

The same existing classifier-fitting function, fitting-only normalization, balanced BCE, optimizer, patience, selection metric and threshold are shared with I-JEPA. Model-native preprocessing and its full-frame geometry control are declared separately. Assessment is excluded from fitting, normalization and checkpoint choice, and requires a sealed verified run. Previously inspected recordings provide retrospective assessment rather than a new blind test.

The selected supervised configuration uses encoder LR `1e-5`, head LR `1e-3`, effective batch 32 (physical batch 8), 20 epochs and one warmup epoch. See [config.json](config.json) for the recipe.

## Results

| Metric | Validation: 6 recordings | Assessment: 9 recordings |
| --- | ---: | ---: |
| Recording-weighted balanced accuracy | 92.55% | 79.57% |
| Defect recall | 88.53% | 81.51% |
| False-alarm rate | 3.43% | 22.38% |
| AUROC | 0.9847 | 0.8683 |
| False-positive frames | 33 | 524 |
| False-negative frames | 390 | 594 |

Each frame has weight `1 / frames_in_its_recording`; class recalls are calculated with those weights and averaged for balanced accuracy. Raw frame counts have a different denominator. Full aggregate metrics and selected checkpoint details are in [results.json](results.json). Assessment was opened only after both new models were selected and reloaded successfully. No threshold tuning, fitting-plus-validation refit or assessment-driven retraining was performed.

## Usage

Run from this model directory:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py fetch-source
python run.py predict --image /absolute/path/weld.png --device cpu
```

## Weights

`weights/best/` contains the encoder and matching classifier used for the reported results. [manifest.json](weights/manifest.json) records downloads, hashes and inference settings.

| File | Download |
| --- | --- |
| `best/encoder.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--encoder.pt) |
| `best/classifier.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--classifier.pt) |


## Implementation and licenses

Inference and preprocessing are in [code/](code/). Read [licenses/](licenses/) for source and checkpoint terms.
