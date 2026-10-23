# DINOv3 ViT-L/16 + linear classifier

## Status

Completed. The evaluated checkpoint is in `weights/best/`. [model.json](model.json), [config.json](config.json), [data.json](data.json) and [results.json](results.json) record the architecture, recipe, shared split and measured results.

## Architecture

Released web-pretrained ViT-L/16: 24 blocks, width 1024, patch size 16. Concatenate the final normalized CLS token with the mean of normalized patch tokens, excluding register tokens, for 2048 features.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The encoder remains fixed and only the linear classifier updates, following the declared frozen-feature transfer approach.

## Data

Exactly the final adapted I-JEPA recording split: fitting **12 recordings / 10,338 frames**, validation **6 / 6,454**, assessment **9 / 6,763**. [data.json](data.json) includes class counts and exact normalized-row hashes. All recording segments remain together; no random frame split or fitting-plus-validation refit is used. These canonical recording roles supersede the original archive folder names; a path containing `train` or `test` is not itself a role assignment for this benchmark.

Dataset: [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`. Assessment recordings were inspected in earlier research, so these scores are retrospective held-out results.

## Training and selection

Use the shared data preparation, classifier recipe and evaluation commands. Raw paths, annotations, labels and checksums must match the [split fingerprints](data.json). Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`.

The same existing classifier-fitting function, fitting-only normalization, balanced BCE, optimizer, patience, selection metric and threshold are shared with I-JEPA. Model-native preprocessing and its full-frame geometry control are declared separately. Assessment is excluded from fitting, normalization and checkpoint choice, and requires a sealed verified run. Previously inspected recordings provide retrospective assessment rather than a new blind test.

## Results

| Metric | Validation: 6 recordings | Assessment: 9 recordings |
| --- | ---: | ---: |
| Recording-weighted balanced accuracy | 89.70% | 88.00% |
| Defect recall | 85.91% | 83.45% |
| False-alarm rate | 6.51% | 7.45% |
| AUROC | 0.9660 | 0.9626 |
| False-positive frames | 119 | 310 |
| False-negative frames | 360 | 658 |

Each frame has weight `1 / frames_in_its_recording`; class recalls are calculated with those weights and averaged for balanced accuracy. Raw frame counts have a different denominator. Full aggregate metrics and selected checkpoint details are in [results.json](results.json). Assessment was opened only after both new models were selected and reloaded successfully. No threshold tuning, fitting-plus-validation refit or assessment-driven retraining was performed.

## Usage

Run from this model directory:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py test
python run.py --help
```

```bash
python run.py prepare --metadata /absolute/path/registration.json \
  --data-root /absolute/path/dataset-root --output runs/prepared
python run.py fetch-assets --output assets/pretrained
python run.py train --data runs/prepared/data.local.json --assets assets/pretrained/assets.local.json \
  --output runs/training --device cuda --batch-size 16 --workers 4
python run.py verify --run runs/training --device cuda
python run.py evaluate --run runs/training --device cuda
```

The model identifier is supplied automatically by this folder’s runner. Asset setup verifies pinned upstream source and checkpoint bytes. The released encoder is available through the public weight downloader; the asset command additionally sets up pinned upstream source. Both selected checkpoints were sealed and replayed before either assessment was opened.


## Weights

`weights/best/` contains the evaluated encoder, classifier and checkpoint descriptor. `weights/final/` retains the last completed epoch of the selected configuration. The assessment uses **best**. The frozen encoder is stored once under `best/` and reused by the final classifier. Actual files and their SHA-256 hashes are listed once in [weights/manifest.json](weights/manifest.json); [weights/README.md](weights/README.md) has the download links. Large encoder bytes are release assets; small classifiers and descriptors are committed here.

```bash
python run.py fetch-weights
python run.py verify-weights
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint best
```

Use `--checkpoint final` to load the retained final checkpoint. GPU scores are reported; CPU/GPU metric equivalence has not been established.

y predict-local --image /absolute/path/weld.png --device cpu --checkpoint best
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint final
```

## Implementation and licenses

Implementation lives in [code/](code/) and tests in [tests/](tests/). This model imports no code from another model folder. Read [licenses/](licenses/) for applicable source and checkpoint terms. Pinned upstream source is checked before loading. Checkpoints retain the exact saved training bytes; package/export code does not retrain or alter tensors. No classifier loss or EMA update is added to the released encoder during frozen-feature training.
