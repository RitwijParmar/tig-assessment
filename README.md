# TIG Assessment

Four weld-classification models using the same recording split and a binary linear classifier.

| Model | Status | Validation BA | Assessment BA | Folder |
| --- | --- | ---: | ---: | --- |
| I-JEPA ViT-H/14 (Pretrained) | Completed | 94.78% | 84.89% | [Model](models/ijepa-vit-h14-pretrained/) |
| I-JEPA ViT-H/14 (Adapted) | Completed | 98.41% | 92.75% | [Model](models/ijepa-vit-h14-adapted/) |
| MobileNetV5-300M | Completed | 92.55% | 79.57% | [Model](models/mobilenetv5-300m/) |
| DINOv3 ViT-L/16 | Completed | 89.70% | 88.00% | [Model](models/dinov3-vit-l16/) |

| Split | Recordings | Frames |
| --- | ---: | ---: |
| Training | 12 | 10,338 |
| Validation | 6 | 6,454 |
| Assessment | 9 | 6,763 |

All four use identical frame memberships and labels. Complete recordings stay in one split. Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083). BA is recording-weighted balanced accuracy at threshold 0.5. Assessment recordings were inspected in earlier research; these are retrospective held-out results.

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
    best/
      encoder.pt          # .safetensors for DINOv3
      classifier.pt
      model.json
    final/
      encoder.pt          # .safetensors for DINOv3
      classifier.pt
      model.json
```

**Best** is validation-selected and supplies the reported results. **Final** is the last downstream training epoch. I-JEPA final classifiers are labelled CPU reconstructions.

Run inside a model folder:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint best
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint final
```

Before native-model inference, set up pinned source: MobileNetV5 uses `python run.py fetch-assets --output assets/pretrained --source-only`; DINOv3 uses `python run.py fetch-assets --output assets/pretrained`.

Each model's weight manifest provides downloads and SHA-256 checks. Training commands, architecture, data and results are in its model card. Validate with `python tools/verify_model_packages.py` and `python tools/check_all_models.py`.
