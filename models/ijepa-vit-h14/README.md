# Weld-adapted I-JEPA ViT-H/14 + linear classifier

## Status

Completed. The evaluated checkpoint is in `weights/best/`. [model.json](model.json), [config.json](config.json), [data.json](data.json) and [results.json](results.json) record the architecture, recipe, shared split and measured results.

## Architecture

**Stage 1 — JEPA adaptation**

```text
Same RGB crop, 224 × 224
  ├─ Visible context patches → Encoder 1 → Predictor → Predicted target embeddings
  └─ All crop patches        → Encoder 2             → Target embeddings
                                                         ↓
                                Smooth L1 loss at the target patch positions
```

| Component | Structure | Weight updates during adaptation |
| --- | --- | --- |
| Encoder 1 | ViT-H/14: 32 blocks, width 1280 | Gradients |
| Predictor | 12 blocks, width 384, 16 attention heads; output width 1280 | Gradients |
| Encoder 2 | ViT-H/14: 32 blocks, width 1280 | EMA of Encoder 1 |

All three load matched roles from the official pretrained I-JEPA checkpoint. There is no LoRA. Fixed sinusoidal positions stay fixed; target-encoder weights change through EMA even though target gradients are disabled.

The image has 256 patches on a 16×16 grid. The context encoder sees only visible patches. The target encoder sees the complete crop; its embeddings are feature-normalized before gathering target positions. The predictor estimates four target blocks. Smooth L1 loss, beta 1, is averaged over blocks, tokens and dimensions, with recording-balanced image weights. The loss does not use weld labels.

**Stage 2 — classifier training and prediction**

```text
Full image → letterbox → adapted Encoder 2 → valid-patch mean pooling
           → fitting-only feature normalization → Linear(1280, 1) → good / defect
```

During this stage, Encoder 2 is held fixed and **only the classifier trains**. Encoder 1 and the predictor are not used. During prediction, all weights are fixed.

## Data

Source: [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083), associated with [Bacioiu et al. (2019)](https://doi.org/10.1016/j.jmapro.2019.07.020).

Released labels: **0** good weld, **1** burn through, **2** contamination, **3** lack of fusion, **4** misalignment, **5** lack of penetration. The binary target is `int(label6 > 0)`.

| Role | Recording groups | Frames | Good | Defect | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| Training / fitting | 12 | 10,338 | 3,915 | 6,423 | Weight updates |
| Validation / selection | 6 | 6,454 | 1,341 | 5,113 | Checkpoint selection |
| Assessment / testing | 9 | 6,763 | 2,627 | 4,136 | Evaluate selected model |

Group by image parent-folder name after removing terminal `-partN`; all segments of that recording stay together. This is a filename-based recording proxy. Roles do not overlap, and six additional reserved groups are excluded. Assessment images do not enter this run's adaptation, classifier fitting, normalization fitting or checkpoint selection. These assessment recordings were inspected in earlier research, so this is not a new blind external test.

JEPA uses the pinned upstream random resized 224px crop, scale 0.3–1.0, and its native four-target mask collator. Context scale is 0.85–1.0; target scale 0.15–0.20, aspect ratio 0.75–1.5. The native retry fallback is preserved. Downstream images use whole-image, centered black letterboxing at 224px, bilinear resize and ImageNet normalization. Mean pooling excludes padding-intersecting patches.

Local metadata records `path, group, frame, label6, target, crc, original_bytes`. Image reads verify containment, byte size and CRC32. `data.json` holds the exact split fingerprints; `config.json` holds the recipe; frame-level metadata and images remain local. Historical `train`/`test` directory names do not assign current roles—the explicit recording split does.

## Training and selection

JEPA: seed 43, five-epoch schedule, batch 16 × accumulation 2, AdamW, context LR 3e-5, predictor LR 1e-4, weight decay 0.04, one warmup epoch then cosine decay, gradient clipping 1.0. EMA momentum rises from 0.996 to 1.0. Biases and one-dimensional parameters have no weight decay.

Classifier: fitting-only feature mean/population standard deviation, zero-initialized linear head, recording/class-balanced BCE, AdamW LR 0.001, weight decay 0.01, batch 256, at most 100 epochs, validation patience 15. Highest validation balanced accuracy selects the checkpoint; earliest exact ties win.

The recorded winner is **adaptation epoch 1, after 324 updates**, with classifier epoch 3. It belongs to a five-epoch cosine/EMA schedule: shortening the schedule to one epoch does not reproduce that checkpoint.

## Results

| Recording-weighted metric | Validation | Assessment |
| --- | ---: | ---: |
| Balanced accuracy | **98.41%** | **92.75%** |
| Defect recall | **97.58%** | **96.51%** |
| False-alarm rate | **0.77%** | **11.00%** |

The threshold is **0.5**. Each frame's metric weight is `1 / all_frames_in_its_recording`; balanced accuracy averages weighted good and defect recall. Assessment has 165 missed defect frames and 521 false-alarm frames. Raw frame counts and recording-weighted percentages use different denominators. The selected checkpoint, hashes and metrics are in [results.json](results.json).

### Known limitation: concentrated false alarms

The adapted I-JEPA ViT-H/14 model fails on many good frames from assessment recording **`170906-141809`** (group `170906-141809-Al 2mm`):

- **1,037 good frames**, of which **514** are incorrectly classified as defective (**49.57%**).
- This one recording contributes **514 of the 521 total assessment false alarms**; the other assessment recordings contribute seven.
- This recording is absent from validation, so validation's **0.77% recording-weighted false-alarm rate did not reveal this failure**.

These counts establish where the errors concentrate. The visual or annotation cause has not been established. The aggregate 11.00% recording-weighted assessment false-alarm rate is therefore not evidence of uniformly low false alarms across recordings.

## Usage

Run from this model directory:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py test
python run.py --help
```

For the retained completed model:

```bash
python run.py predict-local --image /absolute/path/weld.png --device cpu
```

To reproduce fitting using the exact recorded data roles, obtain the original official full pretraining checkpoint separately and run:

```bash
python run.py prepare --metadata /absolute/path/registration.json \
  --data-root /absolute/path/dataset-root \
  --base-checkpoint /absolute/path/IN1K-vit.h.14-300e.pth.tar --output runs/prepared
python run.py train --config runs/prepared/config.local.json --output runs/training --device cuda
python run.py evaluate --run runs/training --device cuda
```


## Weights

`weights/best/` contains the evaluated encoder, classifier and checkpoint descriptor. The historical last-epoch checkpoint was not retained; no substitute final checkpoint is created. Actual files and their SHA-256 hashes are listed once in [weights/manifest.json](weights/manifest.json); [weights/README.md](weights/README.md) has the download links. Large encoder bytes are release assets; small classifiers and descriptors are committed here.

```bash
python run.py fetch-weights
python run.py verify-weights
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint best
```

 inference layout. The small classifiers are also stored directly in Git; large encoders are release assets. MobileNetV5/DINOv3 currently contain pretrained encoders only.

## Implementation and licenses

Implementation lives in [code/](code/) and tests in [tests/](tests/). This model imports no code from another model folder. Read [licenses/](licenses/) for applicable source and checkpoint terms. Numerical training settings, learned tensor values and recorded metrics are preserved; the original encoder export only changes its checkpoint container.
