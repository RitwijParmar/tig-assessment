# Checkpoints

Best is selected on validation and is used for the reported assessment. Both `best/` and `final/` contain an encoder, classifier and `model.json` descriptor. Final means the last completed downstream training epoch of the selected configuration.

The historical probe saved only its best head. `final/classifier.pt` is an explicitly labelled **CPU reconstruction**, at epoch 29, from the original verified fitting/validation features and recipe. Every historical epoch's validation score and the original stopping epoch were reproduced; the replayed best head agrees with the original GPU head within 4.5e-8. Bitwise identity to an unsaved original final head is not claimed. No encoder training or assessment was performed in recovery. The final encoder is identical to best because downstream training freezes it.

```bash
python run.py fetch-weights
python run.py verify-weights
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint final
```

Run from this model directory. Downloads verify every part and reconstructed file against its size and SHA-256. Identical frozen encoder bytes are reused locally.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `best/encoder.pt` | 2523165301 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part02) |
| `best/classifier.pt` | 18633 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--classifier.pt) |
| `final/encoder.pt` | 2523165301 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14-pretrained--encoder.pt.part02) |
| `final/classifier.pt` | 19807 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/ijepa-final-recovery-20261023/ijepa-vit-h14-pretrained--final--classifier.pt) |

[manifest.json](manifest.json) records file hashes and exact pairings. Read the applicable [licenses](../licenses/). Reported benchmark metrics refer to **best**, not final.
