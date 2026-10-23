# Checkpoints

Best is selected on validation and is the checkpoint used for the reported assessment. `weights/final/` retains the last completed epoch of the selected configuration. The assessment uses **best**. The frozen encoder is stored once under `best/` and reused by the final classifier.

```bash
python run.py fetch-weights
python run.py verify-weights
```

Run these commands from this model directory. The downloader places each file at the path below and verifies file/part sizes and SHA-256.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `best/encoder.safetensors` | 1212559808 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors) |
| `best/classifier.pt` | 27785 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--best--classifier.pt) |
| `final/classifier.pt` | 27795 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--final--classifier.pt) |

Multipart encoder slices are reassembled automatically. [manifest.json](manifest.json) records all hashes and the exact encoder/classifier pairing. See [licenses](../licenses/) for the applicable terms.
