# Checkpoints

Best is selected on validation and is used for the reported assessment. Both `best/` and `final/` contain an encoder, classifier and `model.json` descriptor. Final means the last completed downstream training epoch of the selected configuration.

Both encoder and classifier were trained jointly. The original best (epoch 16) and final (epoch 20) checkpoint bytes are unchanged.

```bash
python run.py fetch-weights
python run.py verify-weights
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint final
```

Run from this model directory. Downloads verify every part and reconstructed file against its size and SHA-256. Identical frozen encoder bytes are reused locally.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `pretrained/encoder.safetensors` | 1176564736 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/mobilenetv5-300m--pretrained_encoder.safetensors) |
| `best/encoder.pt` | 1176686195 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--encoder.pt) |
| `best/classifier.pt` | 28027 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--classifier.pt) |
| `final/encoder.pt` | 1176686749 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--final--encoder.pt) |
| `final/classifier.pt` | 28037 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--final--classifier.pt) |

[manifest.json](manifest.json) records file hashes and exact pairings. Read the applicable [licenses](../licenses/). Reported benchmark metrics refer to **best**, not final.
