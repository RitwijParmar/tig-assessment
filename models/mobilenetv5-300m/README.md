# MobileNetV5-300M + linear classifier

## Status

Implementation ready; matched weld training and assessment have not run. [model.json](model.json) gives the common package metadata. [config.json](config.json) pins the released model, preprocessing/readout and fitting recipe. [results.json](results.json) records unavailable metrics explicitly.

This folder is independently runnable. Its implementation, tests, configuration, data identities and checkpoint downloader are all contained here.

## Architecture

Released Gemma 3n vision encoder, including its native multiscale fusion adapter and global pooling. Native 768px evaluation yields a 2048-dimensional image vector.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The primary encoder remains fixed; only the linear classifier updates. A separate optional supervised arm updates the encoder and classifier, retaining the frozen baseline as epoch 0.

## Data

Exactly the final adapted I-JEPA recording split: fitting **12 recordings / 10,338 frames**, validation **6 / 6,454**, assessment **9 / 6,763**. [data.json](data.json) includes class counts and exact normalized-row hashes. All recording segments remain together; no random frame split or fitting-plus-validation refit is used.

Dataset: [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`. Assessment recordings were inspected in earlier research, so future scores remain retrospective held-out results.

## Training and selection

Use the shared data preparation, classifier recipe and evaluation commands. Raw paths, annotations, labels and checksums must match the [split fingerprints](data.json). Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`.

The same existing classifier-fitting function, fitting-only normalization, balanced BCE, optimizer, patience, selection metric and threshold are shared with I-JEPA. Model-native preprocessing and its full-frame geometry control are declared separately. Assessment is excluded from fitting, normalization and checkpoint choice, and requires a sealed verified run. Previously inspected recordings provide retrospective assessment rather than a new blind test.

## Results

No matched weld-classification metrics are available. [results.json](results.json) has null validation and assessment entries. Existing historical scores from other recording splits are not substituted. Real pretrained CPU forward checks passed; these are implementation checks, not weld-classification results.

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
  --output runs/training --device cuda
python run.py verify --run runs/training --device cuda
python run.py evaluate --run runs/training --device cuda
```

The model identifier is supplied automatically by this folder’s runner. Asset setup verifies pinned upstream source and checkpoint bytes. The released encoder is available through the public weight downloader; the asset command additionally sets up pinned upstream source. No assessment result is claimed until matched training and selection complete.


## Weights

Actual checkpoints are available through [weights/README.md](weights/README.md), with direct download links. `python run.py fetch-weights` reconstructs the exact files under this model’s `weights/` directory and checks their SHA-256 hashes. [weights/manifest.json](weights/manifest.json) identifies every part and final file. Both I-JEPA folders use the same `encoder.pt` + `classifier.pt` inference layout. The small classifiers are also stored directly in Git; large encoders are release assets. MobileNetV5/DINOv3 currently contain pretrained encoders only.

## Implementation and licenses

Implementation lives in [code/](code/) and tests in [tests/](tests/). This model imports no code from another model folder. Read [licenses/](licenses/) for applicable source and checkpoint terms. Numerical training settings, learned tensor values and recorded metrics are preserved; the original encoder export only changes its checkpoint container.
