# Weights

`best/` contains the validation-selected encoder, matching classifier and checkpoint descriptor used for the reported results.

```bash
python run.py fetch-weights
python run.py verify-weights
```

| File | Download |
| --- | --- |
| `best/encoder.safetensors` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/dinov3-vit-l16--best--encoder.safetensors) |
| `best/classifier.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/dinov3-vit-l16--best--classifier.pt) |

[manifest.json](manifest.json) records sizes, SHA-256 hashes and checkpoint pairing.
