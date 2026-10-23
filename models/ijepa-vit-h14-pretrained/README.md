# Original pretrained I-JEPA ViT-H/14 + linear classifier

## Status

Completed original-pretrained baseline. The encoder keeps its released pretrained weights and only the linear classifier trains. [model.json](model.json) gives the common package metadata; [config.json](config.json) records this completed control's exact recipe. It is a recorded-recipe file, not a new baseline-only training CLI configuration.

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

[config.json](config.json) records the recipe and exact split fingerprints. [results.json](results.json) records the selected checkpoint identity and exact metrics. Weights, frame-level annotations and predictions remain local.

## Results

| Recording-weighted metric | Validation | Assessment |
| --- | ---: | ---: |
| Balanced accuracy | **94.78%** | **84.89%** |
| Defect recall | **89.72%** | **85.02%** |
| False-alarm rate | **0.16%** | **15.25%** |

Validation contains 229 missed defect frames and 3 false alarms. Assessment contains 747 missed defect frames and 729 false alarms. Each frame's metric weight is `1 / all_frames_in_its_recording`; class recalls are calculated with these weights, then averaged for balanced accuracy. Raw frame counts and recording-weighted percentages therefore have different denominators.

The adapted final model reaches **92.75%** assessment balanced accuracy, **96.51%** defect recall and **11.00%** false-alarm rate on these same nine recordings. These paired observations describe the completed run; they do not establish universal or statistically proven superiority.

## Usage

Run commands from the repository root. Install [requirements-ijepa.txt](../../requirements-ijepa.txt) and run `python -m pytest -q tests/ijepa`. Shared commands are listed by `python -m ijepa --help`.

For the retained completed baseline, load its own files with the shared classifier loader:

```python
import json
from pathlib import Path
from ijepa.classifier import WeldClassifier
from ijepa.evaluate import predict_image

folder = Path("models/ijepa-vit-h14-pretrained")
result = json.loads((folder / "results.json").read_text())
model = WeldClassifier.from_files(
    folder / "weights/pretrained_encoder.pth.tar",
    folder / "weights/classifier.pt",
    device="cpu", encoder_role="target_encoder",
    expected={
        "encoder": result["weights"]["base_sha256"],
        "classifier": result["weights"]["classifier_sha256"],
    },
)
print(predict_image(model, "/absolute/path/weld.png", "cpu"))
```

The shared `train` command fits the original epoch-0 classifier before adaptation, then runs the declared adaptation experiment. It does not expose a separate baseline-only fitting command. This folder's existing results do not require rerunning that experiment.

## Weights

The original checkpoint and this baseline's trained classifier are saved locally under [weights/](weights/). [weights/manifest.json](weights/manifest.json) records relative filenames, sizes and SHA-256 checksums. Use the checkpoint's original `target_encoder` role with this baseline's classifier. Checkpoint bytes remain outside Git.

## Implementation and licenses

The baseline and adapted model share the encoder implementation, preprocessing, readout fitting and metrics. Source lives in [the I-JEPA package](../../ijepa/) rather than being duplicated here:

- [model.py](../../ijepa/model.py): pretrained roles, encoder construction and strict loading.
- [classifier.py](../../ijepa/classifier.py): frozen feature extraction, linear-probe fitting and inference.
- [data.py](../../ijepa/data.py): exact role checks, image integrity and preprocessing.
- [train.py](../../ijepa/train.py): readout fitting at pretrained epoch 0 and subsequent adaptation checkpoints.
- [metrics.py](../../ijepa/metrics.py): recording-weighted evaluation.
- [Tests](../../tests/ijepa/) and [attribution/license terms](../../ijepa/NOTICE.md).


