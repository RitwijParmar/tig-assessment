# TIG Assessment

Models and recording-grouped assessments for weld-image classification on the [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083).

## Models

| Model | Implementation, architecture, data and results |
| --- | --- |
| I-JEPA ViT-H/14 with weld adaptation and a linear classifier | [Model directory](models/ijepa-vit-h14/) |

Each model lives in its own directory under `models/`, including its recipe, source, tests and summary metrics. Additional approaches can be added as separate model directories.

## Evaluation scope

The current model uses separate fitting, validation and assessment recording groups. Assessment recordings are excluded from this run's weight updates but were inspected in earlier research; these are diagnostic held-out results, not a new blind external test.

The current model's false alarms concentrate in recording `170906-141809`: **514 of the 521 assessment false alarms**, among **1,037 good frames** in that recording. See the [model's known limitation](models/ijepa-vit-h14/README.md#known-limitation-concentrated-false-alarms) for details and metric definitions.

This repository publishes source, documentation and summary results. Obtain the dataset and pretrained weights separately; local predictions, checkpoints and frame-level metadata are excluded. Source and pretrained-weight license conditions are documented in the model's [NOTICE](models/ijepa-vit-h14/NOTICE.md).
