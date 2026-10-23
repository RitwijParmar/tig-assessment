# Checkpoints

Best is selected on validation and is the checkpoint used for the reported assessment. `weights/final/` retains the last completed epoch of the selected configuration. The assessment uses **best**. The released base encoder is stored once under `pretrained/` and shared by best/final loading.

```bash
python run.py fetch-weights
python run.py verify-weights
```

Run these commands from this model directory. The downloader places each file at the path below and verifies file/part sizes and SHA-256.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `pretrained/encoder.safetensors` | 1176564736 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/mobilenetv5-300m--pretrained_encoder.safetensors) |
| `best/encoder.pt` | 1176686195 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--encoder.pt) |
| `best/classifier.pt` | 28027 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--classifier.pt) |
| `final/encoder.pt` | 1176686749 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--final--encoder.pt) |
| `final/classifier.pt` | 28037 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--final--classifier.pt) |

Multipart encoder slices are reassembled automatically. [manifest.json](manifest.json) records all hashes and the exact encoder/classifier pairing. See [licenses](../licenses/) for the applicable terms.
