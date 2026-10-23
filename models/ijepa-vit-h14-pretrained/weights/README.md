# Downloadable weights

Actual checkpoint downloads are published in the [model-weight release](https://github.com/RitwijParmar/tig-assessment/releases/tag/model-weights-20261023). The downloader saves and verifies them in this folder.

```bash
python weights/download.py
python weights/download.py --verify-only
```

## selected_encoder

`encoder.pt` — 2523165301 bytes; SHA-256 `2fd11db9cf9ade5f7d707faedb78f3bde64feea79c5d18e5f498f86620bdb671`.

- [Download ijepa-vit-h14-pretrained--encoder.pt.part01](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part01)
- [Download ijepa-vit-h14-pretrained--encoder.pt.part02](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part02)

## classifier

`classifier.pt` — 18633 bytes; SHA-256 `c1b949da5b97acea322495d3aeda835c4c34b8a859812e4eb4bc625f513f3d40`.

- [Download ijepa-vit-h14-pretrained--classifier.pt](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--classifier.pt)

Numbered parts are byte slices of the exact checkpoint, reconstructed automatically and verified individually and as a complete file. Each part stays below GitHub’s release-file limit.

I-JEPA encoders must be paired with the classifier in the same model folder. MobileNetV5/DINOv3 have released pretrained encoders only; matched weld classifiers have not been trained. Original full I-JEPA pretraining checkpoints are optional assets and are not required for inference with the encoder exports. See [model licenses](../licenses/).
