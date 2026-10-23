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
```

## Weights

`weights/best/` contains the encoder and matching classifier used for the reported results. [manifest.json](weights/manifest.json) records downloads, hashes and inference settings.

| File | Download |
| --- | --- |
| `best/encoder.pt` | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--encoder.pt.part02) |
| `best/classifier.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--best--classifier.pt) |

**Note:** Part 1 and part 2 are pieces of one encoder file. The encoder is about 2.52 GB, so it was split to fit GitHub’s release-file limit. `python run.py fetch-weights` automatically joins and verifies them. [GitHub limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).

In the previous layout, the `final/encoder.pt` row linked to `best` because both used identical frozen encoder weights; their classifier weights differed. Only the best checkpoint is distributed here.


## Implementation and licenses

Inference and preprocessing are in [code/](code/). Read [licenses/](licenses/) for source and checkpoint terms.
