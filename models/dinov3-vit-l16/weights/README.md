# Downloadable weights

Actual checkpoint downloads are published in the [model-weight release](https://github.com/RitwijParmar/tig-assessment/releases/tag/model-weights-20261023). The downloader saves and verifies them in this folder.

```bash
python weights/download.py
python weights/download.py --verify-only
```

## pretrained_encoder

`pretrained_encoder.safetensors` — 1212559808 bytes; SHA-256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

- [Download dinov3-vit-l16--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors)

## selected_encoder

`pretrained_encoder.safetensors` — 1212559808 bytes; SHA-256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.

- [Download dinov3-vit-l16--pretrained_encoder.safetensors](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors)

Numbered parts are byte slices of the exact checkpoint, reconstructed automatically and verified individually and as a complete file. Each part stays below GitHub’s release-file limit.

I-JEPA encoders must be paired with the classifier in the same model folder. MobileNetV5/DINOv3 have released pretrained encoders only; matched weld classifiers have not been trained. Original full I-JEPA pretraining checkpoints are optional assets and are not required for inference with the encoder exports. See [model licenses](../licenses/).
