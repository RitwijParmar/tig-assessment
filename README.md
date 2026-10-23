# TIG Assessment

Four weld-classification models using the same recording split and a binary linear classifier.

| Model | Status | Validation BA | Assessment BA | Folder |
| --- | --- | ---: | ---: | --- |
| I-JEPA ViT-H/14 (Pretrained) | Completed | 94.78% | 84.89% | [Model](models/ijepa-vit-h14-pretrained/) |
| I-JEPA ViT-H/14 (Adapted) | Completed | 98.41% | 92.75% | [Model](models/ijepa-vit-h14-adapted/) |
| MobileNetV5-300M | Completed | 92.55% | 79.57% | [Model](models/mobilenetv5-300m/) |
| DINOv3 ViT-L/16 | Completed | 89.70% | 88.00% | [Model](models/dinov3-vit-l16/) |

**Adapted I-JEPA false-alarm finding:** The increase from **0.77% on validation to 11.00% on assessment** is concentrated in recording **`170906-141809`**, which accounts for **514 of 521 assessment false alarms (98.66%)**. The other eight assessment recordings contribute just **seven**. This recording is absent from validation, so its failure was not reflected in the validation rate. See the [recording-level breakdown](models/ijepa-vit-h14-adapted/README.md#known-limitation-concentrated-false-alarms). The reported assessment results include all nine recordings.

| Split | Recordings | Frames |
| --- | ---: | ---: |
| Training | 12 | 10,338 |
| Validation | 6 | 6,454 |
| Assessment | 9 | 6,763 |

All four use identical frame memberships and labels. Complete recordings stay in one split. Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083). BA is recording-weighted balanced accuracy at threshold 0.5. Assessment recordings were inspected in earlier research; these are retrospective held-out results.

```text
models/<model>/
  README.md
  config.json
  data.json
  results.json
  requirements.txt
  run.py
  code/                  # training, evaluation and inference
  tests/
  licenses/
  weights/
    README.md            # download links and multipart instructions
    manifest.json
    best/
      encoder.pt          # .safetensors for DINOv3
      classifier.pt
```

Only the validation-selected **best** checkpoint is distributed for each model. It supplies the reported results.

Run inside a model folder:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py fetch-source
python run.py predict --image /absolute/path/weld.png --device cpu
```

Each model folder contains its architecture, data split, training recipe, results, best weights and training, evaluation and inference code. Weight downloads verify all SHA-256 hashes. Native encoder source is pinned and downloaded by `fetch-source`; I-JEPA source is bundled.

**I-JEPA download note:** Part 1 and part 2 are pieces of one encoder file. The encoder is about 2.52 GB, so it was split to fit GitHub’s release-file limit. `python run.py fetch-weights` automatically joins and verifies them. [GitHub limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).

Each model README includes complete `prepare`, `train` and `evaluate` commands. `evaluate --run` assesses a sealed training run; `evaluate-best --data ... --output ...` assesses the published best pair directly using the exact prepared split. Neither evaluation path fits a classifier or tunes a threshold. Use `python run.py test` inside each model to check training and checkpoint isolation.

Check package metadata with `python verify.py`; use `python verify.py --weights` after downloading checkpoints.
