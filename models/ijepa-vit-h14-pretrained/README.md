# I-JEPA ViT-H/14 (Pretrained)

## Status

Completed. The evaluated checkpoint is in `weights/best/`. [config.json](config.json), [data.json](data.json) and [results.json](results.json) record the architecture, recipe, shared split and measured results.

## Architecture

```text
Full RGB image → letterbox 224 × 224 → original pretrained target encoder
              → valid-patch mean pooling → fitting-only feature normalization
              → Linear(1280, 1) → sigmoid → good / defect at threshold 0.5
```

The encoder is the official I-JEPA **ViT-H/14 target encoder**: 32 transformer blocks, width 1280, and 14 × 14 pixel patches. A 224 × 224 image produces 256 patch tokens. The encoder retains its original pretrained weights throughout this baseline. Only the binary classifier trains on weld labels. There is no weld JEPA adaptation, predictor loss, EMA update or LoRA in this baseline stage.

The official pretrained checkpoint is [IN1K-vit.h.14-300e.pth.tar](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar), SHA-256 `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`. This baseline uses its `target_encoder` role. Do not substitute the adapted encoder or pair the baseline encoder with the adapted model's classifier.

## Data

Source: [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083), associated with [Bacioiu et al. (2019)](https://doi.org/10.1016/j.jmapro.2019.07.020). The binary label is `int(label6 > 0)`; label 0 is good and labels 1–5 are defects.

| Role | Recording groups | Frames | Good | Defect |
| --- | ---: | ---: | ---: | ---: |
| Training / fitting | 12 | 10,338 | 3,915 | 6,423 |
| Validation / selection | 6 | 6,454 | 1,341 | 5,113 |
| Assessment / testing | 9 | 6,763 | 2,627 | 4,136 |

These are **identical frame memberships and annotations** to the adapted model's roles. All segments of a recording stay together after removing terminal `-partN` from the image parent-folder name. Roles do not overlap; six additional reserved groups are excluded. Group identity is a filename-based recording proxy. Assessment frames do not fit the classifier or normalizer and do not select its checkpoint. These assessment recordings were inspected in earlier research, so this is a retrospective held-out assessment rather than a new blind test.

## Training and selection

The downstream image/readout recipe matches the adapted model: whole-image centered black letterboxing at 224px, bilinear resize, RGB and ImageNet normalization; mean pooling excludes padding-intersecting patches. Feature mean and population standard deviation are fitted on training features only, with standard deviation clamped at 1e-6.

Train a zero-initialized `Linear(1280,1)` using fitting-only recording/class-balanced BCE, AdamW LR 0.001, weight decay 0.01, batch 256, gradient clipping 1.0 and seed 43. The maximum is 100 epochs with validation patience 15; highest validation recording-balanced accuracy at threshold 0.5 selects the checkpoint, with earliest exact ties retained. The recorded classifier was selected at **epoch 14**. The original pretrained encoder remains fixed throughout.

[config.json](config.json) records the recipe; [data.json](data.json) records exact split fingerprints. [results.json](results.json) records the selected checkpoint identity and exact metrics. Weights are downloadable from this folder’s release links; frame-level annotations and predictions remain local.

## Results

| Recording-weighted metric | Validation | Assessment |
| --- | ---: | ---: |
| Balanced accuracy | **94.78%** | **84.89%** |
| Defect recall | **89.72%** | **85.02%** |
| False-alarm rate | **0.16%** | **15.25%** |

Validation contains 229 missed defect frames and 3 false alarms. Assessment contains 747 missed defect frames and 729 false alarms. Each frame's metric weight is `1 / all_frames_in_its_recording`; class recalls are calculated with these weights, then averaged for balanced accuracy. Raw frame counts and recording-weighted percentages therefore have different denominators.

## Usage

Run from this model directory:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py fetch-source
python run.py predict --image /absolute/path/weld.png --device cpu
python run.py test
```

### Training and evaluation

Use canonical metadata with `path, group, frame, label6, target, crc, original_bytes` columns (CSV), or a registration JSON containing `rows`. Paths are relative to the dataset root. Missing CRC/size fields can be computed from local images. `prepare` verifies all recording roles and normalized-row hashes against [data.json](data.json); it does not generate a new random split. Acquire the images and annotations from the dataset linked above.

Training requires the [official full I-JEPA pretraining checkpoint](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar), containing the context encoder, target encoder and predictor. Its SHA-256 is checked against [config.json](config.json). The submitted best encoder alone does not contain all three roles.

```bash
python run.py prepare --metadata /absolute/path/registration.json \
  --data-root /absolute/path/dataset-root \
  --base-checkpoint /absolute/path/IN1K-vit.h.14-300e.pth.tar --output runs/prepared
python run.py train --config runs/prepared/config.local.json --output runs/training --device cuda
python run.py evaluate --run runs/training --device cuda
```

This profile skips JEPA adaptation, keeps the original pretrained target encoder fixed and trains the linear classifier.

To evaluate the **downloaded best weights** directly after data preparation, without retraining:

```bash
python run.py evaluate-best --data runs/prepared/config.local.json \
  --output runs/best-assessment --device cuda --batch-size 2
```

Both evaluation paths use the registered assessment recordings, the saved fitting-only normalization and threshold 0.5. They save aggregate metrics and per-frame predictions locally in a fresh output directory. Training uses fitting images; validation selects the checkpoint. Assessment labels never update weights or choose the checkpoint.

## Weights

Only the validation-selected encoder and matching classifier are distributed in `weights/best/`. See [weights/README.md](weights/README.md) for every file, public download link, multipart instructions and checksum verification. [manifest.json](weights/manifest.json) records the exact pair and inference settings.

## Implementation and licenses

[Training](code/ijepa/train.py), [sealed-run evaluation](code/ijepa/evaluate.py), [best-weight evaluation](code/evaluation.py).

[code/](code/) contains training, preprocessing, checkpoint selection, evaluation and best-checkpoint inference. These paths share the same model and classifier implementation. [tests/](tests/) checks weight updates, split isolation and saved-checkpoint reproduction. Each model folder runs independently. Read [licenses/](licenses/) for source and checkpoint terms.
