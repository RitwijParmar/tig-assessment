# I-JEPA ViT-H/14 (Adapted)

## Status

Completed. The evaluated checkpoint is in `weights/best/`. [config.json](config.json), [data.json](data.json) and [results.json](results.json) record the architecture, recipe, shared split and measured results.

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

The recorded run used `path, group, frame, label6, target, crc, original_bytes` metadata and verified image containment, byte size and CRC32. `data.json` holds the exact split fingerprints; `config.json` holds the recipe; frame-level metadata and images remain local. Historical `train`/`test` directory names do not assign current roles—the explicit recording split does.

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

The increase from **0.77% validation false-alarm rate to 11.00% assessment false-alarm rate** is concentrated in one assessment recording: **`170906-141809`** (group `170906-141809-Al 2mm`). It contributes **514 of the 521 assessment false alarms (98.66%)**.

| Assessment recordings | Good frames | False-alarm frames | Raw good-frame false-alarm rate |
| --- | ---: | ---: | ---: |
| `170906-141809` | 1,037 | 514 | 49.57% |
| Other eight recordings combined | 1,590 | 7 | 0.44% |

False alarms are therefore rare across the remaining eight recordings, while this single-recording failure largely explains the aggregate increase. The recording is absent from validation, so validation's **0.77% recording-weighted false-alarm rate did not reveal it**.

The table uses raw frame counts; the reported **11.00%** uses recording weights. All nine recordings remain included in the reported assessment results. These counts locate the failure; its underlying visual or annotation cause has not been established.

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

This profile runs the complete five-epoch JEPA schedule and fits downstream heads for validation selection. The context encoder and predictor receive gradients; the target encoder follows EMA. The selected target encoder is then fixed for classifier fitting.

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
