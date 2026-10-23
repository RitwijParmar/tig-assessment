# Weights

`best/` contains the validation-selected encoder and its matching classifier used for the reported results. Both are required. The classifiers are committed in this repository; large encoder files are public release downloads.

Run from the model directory:

```bash
python run.py fetch-weights
python run.py verify-weights
```

| Local file | Bytes | Download |
| --- | ---: | --- |
| `best/encoder.pt` | 1,176,686,195 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--encoder.pt) |
| `best/classifier.pt` | 28,027 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--classifier.pt) |

[manifest.json](manifest.json) is the single source for the encoder/classifier pair, SHA-256 checksums, sizes, part checksums and download URLs. Keep the two files from the same model together.

The best inference encoder is not a replacement for the original full pretraining checkpoint when reproducing training. Training prerequisites are documented in the [model README](../README.md).
