# TIG Assessment

Four completed weld-classification models evaluated on the same recording split. Each model contains its own code, tests, configuration, results and checkpoint manifest.

| Model | Status | Validation BA | Assessment BA | Folder |
| --- | --- | ---: | ---: | --- |
| Original pretrained I-JEPA ViT-H/14 + linear classifier | Completed | 94.78% | 84.89% | [Model](models/ijepa-vit-h14-pretrained/) |
| Weld-adapted I-JEPA ViT-H/14 + linear classifier | Completed | 98.41% | 92.75% | [Model](models/ijepa-vit-h14/) |
| MobileNetV5-300M + linear classifier | Completed | 92.55% | 79.57% | [Model](models/mobilenetv5-300m/) |
| DINOv3 ViT-L/16 + linear classifier | Completed | 89.70% | 88.00% | [Model](models/dinov3-vit-l16/) |

Status means training and assessment are complete. Encoder training methods differ and are described in each model card.

| Split | Recordings | Frames |
| --- | ---: | ---: |
| Training | 12 | 10,338 |
| Validation | 6 | 6,454 |
| Assessment | 9 | 6,763 |

All four use identical frame identities, labels and recording groups for each role. [Split fingerprints](models/ijepa-vit-h14/data.json) identify the exact memberships. Entire recordings remain in one role. The dataset is [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083), with binary target `int(label6 > 0)`.

BA is recording-weighted balanced accuracy at threshold 0.5. Assessment was excluded from fitting and checkpoint selection, but these recordings had been inspected in earlier research: these are retrospective held-out results. Different pretraining, native input geometry and encoder-training recipes make this a comparison of complete pipelines.

Every model uses this layout:

```text
models/<model>/
  README.md
  model.json
  config.json
  data.json
  results.json
  requirements.txt
  run.py
  code/
  tests/
  licenses/
  weights/
    README.md
    manifest.json
    download.py
    best/          Evaluated encoder, classifier and model.json
    final/         Last-epoch checkpoint, when retained
    pretrained/    Shared released base, only when required
```

All models have **best**. MobileNetV5 and DINOv3 also retain **final**. Historical I-JEPA final-epoch weights were not retained; they are not replaced with copies of best. Frozen DINOv3 reuses one encoder for both heads. The manifests contain only actual files, each listed once.

Run from any model folder:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py test
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint best
```

MobileNetV5 and DINOv3 additionally require `python run.py fetch-assets --output assets/pretrained` to prepare their pinned upstream source. Model cards document training and assessment commands. Large encoder files live in the [original checkpoint release](https://github.com/RitwijParmar/tig-assessment/releases/tag/model-weights-20261023) and [matched-training release](https://github.com/RitwijParmar/tig-assessment/releases/tag/matched-training-20261023). The downloader stores them in the matching model folder and checks all hashes. Dataset images, frame predictions and optimizer states are excluded from GitHub.

Check all packages with `python tools/verify_model_packages.py` (`--require-weights` verifies local checkpoint bytes). Run the independent model test suites with `python tools/check_all_models.py`.

The adapted I-JEPA false-alarm concentration remains documented in its [model card](models/ijepa-vit-h14/README.md#known-limitation-concentrated-false-alarms): one assessment recording contributed 514 of 521 false alarms. Source and checkpoint license terms are retained under each model's `licenses/`.
