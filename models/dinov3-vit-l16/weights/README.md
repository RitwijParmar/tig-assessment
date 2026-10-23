# Checkpoints

Best is selected on validation and is used for the reported assessment. Both `best/` and `final/` contain an encoder, classifier and `model.json` descriptor. Final means the last completed downstream training epoch of the selected configuration.

Best and final use the same frozen encoder bytes. The downloader creates a hardlink in each folder, avoiding an additional download or duplicate disk allocation.

```bash
python run.py fetch-weights
python run.py verify-weights
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint final
```

Run from this model directory. Downloads verify every part and reconstructed file against its size and SHA-256. Identical frozen encoder bytes are reused locally.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `best/encoder.safetensors` | 1212559808 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors) |
| `best/classifier.pt` | 27785 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--best--classifier.pt) |
| `final/classifier.pt` | 27795 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--final--classifier.pt) |
| `final/encoder.safetensors` | 1212559808 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--pretrained_encoder.safetensors) |

[manifest.json](manifest.json) records file hashes and exact pairings. Read the applicable [licenses](../licenses/). Reported benchmark metrics refer to **best**, not final.
