# Weights

`best/` contains the validation-selected encoder, matching classifier and checkpoint descriptor used for the reported results.

```bash
python run.py fetch-weights
python run.py verify-weights
```

| File | Download |
| --- | --- |
| `best/encoder.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--encoder.pt) |
| `best/classifier.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/matched-training-20261023/mobilenetv5-300m--best--classifier.pt) |

[manifest.json](manifest.json) records sizes, SHA-256 hashes and checkpoint pairing.
