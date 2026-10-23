# Downloadable weights

Actual checkpoint downloads are published in the [model-weight release](https://github.com/RitwijParmar/tig-assessment/releases/tag/model-weights-20261023). The downloader saves and verifies them in this folder.

```bash
python weights/download.py
python weights/download.py --verify-only
```

## selected_encoder

`encoder.pt` — 2523197455 bytes; SHA-256 `9f9c3dfe31e49c1a7333ac85e436d6a7b9f7a12c3a4b142ee12b662a7cde5b17`.

- [Download ijepa-vit-h14--encoder.pt.part01](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--encoder.pt.part01)
- [Download ijepa-vit-h14--encoder.pt.part02](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--encoder.pt.part02)

## classifier

`classifier.pt` — 18633 bytes; SHA-256 `990833bb02a531cb2350b65cb4141630b721d79eed62b9b5cc11b245f93619bb`.

- [Download ijepa-vit-h14--classifier.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--classifier.pt)

Numbered parts are byte slices of the exact checkpoint, reconstructed automatically and verified individually and as a complete file. Each part stays below GitHub’s release-file limit.

I-JEPA encoders must be paired with the classifier in the same model folder. MobileNetV5/DINOv3 have released pretrained encoders only; matched weld classifiers have not been trained. Original full I-JEPA pretraining checkpoints are optional assets and are not required for inference with the encoder exports. See [model licenses](../licenses/).
