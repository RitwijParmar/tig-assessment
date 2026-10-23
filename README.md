# TIG Assessment

Models and recording-grouped assessments for weld-image classification on the [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083).

## Models

| Model | Role | Validation BA | Assessment BA | Details |
| --- | --- | ---: | ---: | --- |
| Original pretrained I-JEPA ViT-H/14 + linear classifier | Completed baseline | **94.78%** | **84.89%** | [Model directory](models/ijepa-vit-h14-pretrained/) |
| Weld-adapted I-JEPA ViT-H/14 + linear classifier | **Selected final model** | **98.41%** | **92.75%** | [Model directory](models/ijepa-vit-h14/) |
| MobileNetV5-300M + linear classifier | Implemented; matched training not run | — | — | [Model directory](models/mobilenetv5-300m/) |
| DINOv3 ViT-L/16 + linear classifier | Implemented; matched training not run | — | — | [Model directory](models/dinov3-vit-l16/) |

The two I-JEPA models were already completed using identical fitting, validation and assessment frame memberships, downstream preprocessing, classifier recipes and threshold 0.5. Training uses **12 recordings / 10,338 frames**, validation **6 / 6,454**, and assessment **9 / 6,763**. BA means recording-weighted balanced accuracy. These entries document completed results; no new experiment is required to obtain them.

Every model directory has the same structure and model-card sections:

```text
models/<model-id>/
  README.md           Status, architecture, data, training, results, usage, weights, licenses
  model.json          Common model metadata and artifact references
  config.json         Model-specific exact configuration or recorded recipe
  results.json        Completed metrics, or explicit nulls when training has not run
  weights/
    README.md         Weight roles and loading notes
    manifest.json     Relative filenames, SHA-256, sizes and availability
    <checkpoint>      Local only; excluded from Git
```

Shared implementation lives in `ijepa/` and `tig_benchmark/`; tests live in `tests/`. Both I-JEPA entries use the same shared package. Run commands from the repository root. Each model's `model.json` identifies its implementation and configuration; configuration contents differ where the model's training differs.

Check all model folders with `python tools/verify_model_packages.py`. Add `--require-weights` to verify every retained local checkpoint's bytes against its SHA-256 inventory. Public clones contain the inventories; local packages also contain the available checkpoint files.

The new MobileNetV5/DINOv3 pipelines use the same exact split and shared readout recipe. See [setup, verification and execution commands](tig_benchmark/README.md). Their quality metrics will be added only after matched runs complete.

## Evaluation scope

All pipelines use separate fitting, validation and assessment recording groups. Assessment recordings are excluded from this run's weight updates but were inspected in earlier research; these are diagnostic held-out results, not a new blind external test.

The selected adapted model's false alarms concentrate in recording `170906-141809`: **514 of the 521 assessment false alarms**, among **1,037 good frames** in that recording. See the [model's known limitation](models/ijepa-vit-h14/README.md#known-limitation-concentrated-false-alarms) for details and metric definitions.

This repository publishes source, documentation and summary results. Obtain the dataset and pretrained weights separately; local predictions, checkpoints and frame-level metadata are excluded. Source and pretrained-weight license conditions are documented in the shared [NOTICE](ijepa/NOTICE.md).
