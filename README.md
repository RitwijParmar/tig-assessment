# TIG Assessment

Each model is an independent directory containing its implementation, tests, data identities, recipe, results and downloadable checkpoints. No model implementation lives at the repository root or imports code from a sibling model directory.

| Model | Status | Validation BA | Assessment BA | Folder |
| --- | --- | ---: | ---: | --- |
| Original pretrained I-JEPA ViT-H/14 + linear classifier | Completed baseline | 94.78% | 84.89% | [Original I-JEPA](models/ijepa-vit-h14-pretrained/) |
| Weld-adapted I-JEPA ViT-H/14 + linear classifier | Selected final model | 98.41% | 92.75% | [Adapted I-JEPA](models/ijepa-vit-h14/) |
| MobileNetV5-300M + linear classifier | Implemented; matched weld training pending | — | — | [MobileNetV5](models/mobilenetv5-300m/) |
| DINOv3 ViT-L/16 + linear classifier | Implemented; matched weld training pending | — | — | [DINOv3](models/dinov3-vit-l16/) |

Every model follows the same layout:

```text
models/<model>/
  README.md          Model card and runnable commands
  model.json         Architecture, status and package references
  config.json        Exact model/training configuration
  data.json          Exact shared recording/frame split fingerprints
  results.json       Recorded metrics, or explicit unavailable results
  requirements.txt   This model's dependencies
  run.py             This model's command-line entry point
  code/              Implementation used by this model
  tests/             Tests runnable independently
  licenses/          Source and weight license terms
  weights/
    README.md        Direct checkpoint download links
    manifest.json    File/part identities, sizes and SHA-256
    download.py      Download, reconstruct and verify checkpoint bytes
    encoder.pt       I-JEPA encoder after download
    classifier.pt    Matching completed I-JEPA classifier
```

**Actual checkpoint files:** [download the model weights](https://github.com/RitwijParmar/tig-assessment/releases/tag/model-weights-20261023). Large encoders are GitHub release assets; each model's `weights/` folder links to its exact files. Both small I-JEPA classifiers are also stored directly in their model folders. The original and adapted I-JEPA packages use the same `encoder.pt` + `classifier.pt` inference layout. MobileNetV5 and DINOv3 currently provide their released pretrained encoders only.

For example:

```bash
cd models/ijepa-vit-h14
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py test
python run.py predict-local --image /absolute/path/weld.png --device cpu
```

Use the same `fetch-weights`, `verify-weights`, `test` and `--help` commands from any model folder. Model-specific preparation, training and assessment commands are documented in each model card. Check package structure with `python tools/verify_model_packages.py`; add `--require-weights` to check all local checkpoint bytes. Run all independent test suites with `python tools/check_all_models.py`.

The dataset is [TIG Aluminium 5083 HDR camera images](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083). All four configurations use exactly **12 fitting recordings / 10,338 frames**, **6 validation recordings / 6,454 frames**, and **9 assessment recordings / 6,763 frames**. The binary target is `int(label6 > 0)`. Frame-level metadata, predictions and images are not published.

BA is recording-weighted balanced accuracy. Assessment recordings were excluded from the completed run's fitting and selection, but had been inspected in earlier research; these are retrospective held-out results. The adapted model's assessment false alarms concentrate in recording `170906-141809`: 514 of 521 false alarms, among 1,037 good frames. The [adapted model card](models/ijepa-vit-h14/README.md#known-limitation-concentrated-false-alarms) preserves that limitation.

Source and checkpoint licenses travel with each model under `licenses/`. The original I-JEPA encoder export preserves every original target-encoder tensor; its container was normalized for consistent loading. No new weld training or assessment was performed during this packaging change.
