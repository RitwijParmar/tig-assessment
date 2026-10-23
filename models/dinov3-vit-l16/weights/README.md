# Weights

`best/` contains the validation-selected encoder and its matching classifier used for the reported results. Both are required. The classifiers are committed in this repository; large encoder files are public release downloads.

Run from the model directory:

```bash
python run.py fetch-weights
python run.py verify-weights
```

| Local file | Bytes | Download |
| --- | ---: | --- |
| `best/encoder.safetensors` | 1,212,559,808 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--best--encoder.safetensors) |
| `best/classifier.pt` | 27,785 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--best--classifier.pt) |

[manifest.json](manifest.json) is the single source for the encoder/classifier pair, SHA-256 checksums, sizes, part checksums and download URLs. Keep the two files from the same model together.

The encoder is the original released pretrained DINOv3 weight; classifier training keeps it fixed. Training prerequisites are documented in the [model README](../README.md).
