# Weights

`best/`: validation-selected checkpoint. `final/`: last downstream training epoch.

Final classifier: reconstructed CPU replay.

```bash
python run.py fetch-weights
python run.py verify-weights
```

| File | Download |
| --- | --- |
| `best/encoder.pt` | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-adapted--best--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-adapted--best--encoder.pt.part02) |
| `best/classifier.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-adapted--best--classifier.pt) |
| `final/encoder.pt` | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-adapted--best--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-adapted--best--encoder.pt.part02) |
| `final/classifier.pt` | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/ijepa-final-recovery-20261023/ijepa-vit-h14-adapted--final--classifier.pt) |

[manifest.json](manifest.json) records sizes, SHA-256 hashes and checkpoint pairings.
