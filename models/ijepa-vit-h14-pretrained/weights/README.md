# Checkpoints

Best is selected on validation and is the checkpoint used for the reported assessment. The historical last-epoch checkpoint was not retained; no substitute final checkpoint is created.

```bash
python run.py fetch-weights
python run.py verify-weights
```

Run these commands from this model directory. The downloader places each file at the path below and verifies file/part sizes and SHA-256.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `best/encoder.pt` | 2523165301 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part02) |
| `best/classifier.pt` | 18633 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--classifier.pt) |

Multipart encoder slices are reassembled automatically. [manifest.json](manifest.json) records all hashes and the exact encoder/classifier pairing. See [licenses](../licenses/) for the applicable terms.
