# Matched MobileNetV5 and DINOv3 transfer pipelines

Executable implementations using the **exact final adapted I-JEPA split**: 12 fitting recordings / 10,338 frames; 6 validation recordings / 6,454 frames; 9 assessment recordings / 6,763 frames. Every frame identity, annotation, checksum and recording role is checked against [the retained split fingerprints](../models/ijepa-vit-h14/config.json). Neither the dataset's folder names nor a newly generated random split assigns roles.

The encoders are released upstream implementations with verified pretrained weights. The binary readout uses the existing I-JEPA [`fit_probe`](../ijepa/classifier.py), [`class_weights` and metrics](../ijepa/metrics.py) directly. The same fitting-only mean/population standard deviation, zero-initialized linear head, balanced BCE, AdamW LR 0.001, weight decay 0.01, batch 256, maximum 100 epochs, patience 15, seed 43 and earliest-tie validation selection apply. The threshold stays 0.5. Validation never joins the fitting set.

## Setup and exact data preparation

Use Python 3.12+ and install [requirements-benchmark.txt](../requirements-benchmark.txt) with torch/torchvision wheels matching your CPU or CUDA platform. Run all commands from the repository root. Source is downloaded only by the explicit `fetch-assets` command; `train`, `evaluate` and `predict` use local registered assets and do not fetch code or weights. Training requires the dataset and its canonical annotation CSV or retained original registration JSON.

```bash
python -m pip install -r requirements-benchmark.txt
python -m tig_benchmark prepare \
  --metadata /absolute/path/registration.json \
  --data-root /absolute/path/dataset-root \
  --output runs/matched-data
```

Preparation derives the exact existing role memberships and rejects changed paths, labels or checksums. The private manifest contains frame-level data and stays outside Git. A registration with the original 27-group cohort or a canonical CSV with the same rows works; source checksum fields can be derived from local images if omitted.

## Assets

```bash
python -m tig_benchmark fetch-assets --model mobilenetv5 --output assets/mobilenetv5

# DINOv3 weights require your approved Hugging Face account.
# Authenticate with hf auth login; never put a token in source or CLI arguments.
python -m tig_benchmark fetch-assets --model dinov3 --output assets/dinov3
```

`fetch-assets` obtains the pinned upstream archive, rejects unsafe archive paths, verifies all upstream Python files against Git blob hashes and verifies the exact released checkpoint SHA-256. Asset registration records paths and identities in `assets.local.json`. To use an existing verified copy instead:

```bash
python -m tig_benchmark register-assets --model dinov3 \
  --source /absolute/path/pinned-dinov3-source \
  --checkpoint /absolute/path/model.safetensors \
  --output assets/dinov3.assets.local.json
```

Upstream source/weight pins are in the model configs and [source integrity manifest](upstream_sources.json). Importing a different already-loaded `timm`/`dinov3` package fails rather than silently selecting it.

## Fit, seal and assess

```bash
python -m tig_benchmark train --model mobilenetv5 \
  --data runs/matched-data/data.local.json \
  --assets assets/mobilenetv5/assets.local.json \
  --output runs/mobilenetv5-native --device cuda --batch-size 2 --workers 4

python -m tig_benchmark train --model dinov3 \
  --data runs/matched-data/data.local.json \
  --assets assets/dinov3/assets.local.json \
  --output runs/dinov3-native --device cuda --batch-size 16 --workers 4

python -m tig_benchmark verify --run runs/mobilenetv5-native --device cuda
python -m tig_benchmark evaluate --run runs/mobilenetv5-native --device cuda
python -m tig_benchmark evaluate --run runs/dinov3-native --device cuda
python -m tig_benchmark predict --run runs/dinov3-native \
  --image /absolute/path/weld.png --device cpu
```

Training reads only fitting and validation images. It verifies frozen encoder parameters/buffers, selects the readout using validation, hashes the chosen weights, data roles, implementation and asset identities, and replays raw fitting images and validation decisions after checkpoint reload. Assessment requires the completed selection/reload status and is a separate command. Repeated assessment and run overwrite are rejected. A failed assessment retains its start marker; use its original artifacts for diagnosis rather than silently rerunning it.

Each run exports the selected final model under `selected/`: `pretrained_encoder.safetensors`, `classifier.pt` (including fitting-only normalization), `model.json`, and `encoder.pt` when supervised encoder changes were selected. The retained files survive download-cache cleanup. Selected versus final-epoch supervised weights are kept separately. The run also stores private pooled features, fitting-only replay fixtures, hashes and metrics. Large encoder weights, images, predictions and optimizer artifacts are not committed. Feature caches are local to a fresh run and include the model/data/geometry identities.

## Native input versus matched geometry

- MobileNetV5: checkpoint-resolved 768px bicubic center-crop evaluation transform, RGB, float pixels in [0,1], zero mean/unit standard deviation. Preserve the built-in multiscale fusion adapter and native pooling; output 2048 features.
- DINOv3 ViT-L/16: official 256px short-side resize and 224px center crop, bicubic/ImageNet normalization. Use Meta's last-block normalized CLS token plus normalized patch-token mean, excluding four register tokens; output 2048 features.

RGB conversion precedes either native transform, including for grayscale source images. These model-native pipelines assess the same frames at different input resolutions/fields of view. They compare transfer pipelines and do not isolate architecture or pretraining data. `--geometry letterbox224` is the separate predeclared full-frame geometry control, retaining each checkpoint's normalization. It uses MobileNetV5 at a non-default resolution and is labeled accordingly. Do not choose native versus common geometry based on assessment results.

No random crop augmentation, CutMix, erasing, LoRA or multi-model fusion is added to the shared readout. This is a declared weld-transfer recipe, not a reproduction of ImageNet or private multimodal pretraining.

## Optional MobileNet full fine-tuning

Use a **new run directory** and add `--finetune` to MobileNet training. It first fits the same frozen baseline, then runs the three declared encoder LRs (1e-5, 3e-5, 1e-4), head LR 0.001, AdamW weight decay 0.01, effective batch 32, gradient clipping 1, a 20-epoch horizon with one warmup epoch and cosine decay, and patience 5. Choose a physical batch dividing 32 based on a fitting-only memory pilot. Normalization stays fitting-only. All models and optimizers restart per candidate; epoch 0 remains eligible, earliest ties retain the frozen baseline, and selected and final checkpoints are separate. The family winner is selected only on validation. This optional supervised recipe is task-specific and has a larger tuning budget than a frozen linear readout.

DINOv3 remains frozen in the declared implementation. Additional folds/seeds are not silently enabled: these commands use only the final I-JEPA split and seed 43 requested here.

## Verification and result status

Run `python -m pytest -q tests` from the repository root. The implementation and model-folder consistency checks are covered by the repository test suite. Tests exercise real image loading with small fixture encoders, role isolation, fitting-only statistics, sealing, tamper rejection, warmup, epoch-0 selection and durable selected-weight exports. Real released-model forward parity was also verified separately on CPU; DINO checkpoint conversion was independently checked against the official Hugging Face model, with maximum feature error 2.5034e-6 (tolerance 1e-4).

The implementation is ready for matched training. **No completed MobileNetV5 or DINOv3 quality result is claimed here.** The existing I-JEPA results remain unchanged. Assessment recordings were historically inspected; future scores on this split remain retrospective held-out results.
