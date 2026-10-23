# Checkpoints

Best is selected on validation and is used for the reported assessment. Both `best/` and `final/` contain an encoder, classifier and `model.json` descriptor. Final means the last completed downstream training epoch of the selected configuration.

The historical probe saved only its best head. `final/classifier.pt` is an explicitly labelled **CPU reconstruction**, at epoch 18, from the original verified fitting/validation features and recipe. Every historical epoch's validation score and the original stopping epoch were reproduced; the replayed best head agrees with the original GPU head within 4.5e-8. Bitwise identity to an unsaved original final head is not claimed. No encoder training or assessment was performed in recovery. The final encoder is identical to best because downstream training freezes it.

The independently archived **epoch-5 JEPA adaptation encoder** is a different training-stage checkpoint. It is recovered separately and must not be paired with this epoch-1 encoder's head. Its recovery is documented in the [recovery release](https://github.com/RitwijParmar/tig-assessment/releases/tag/ijepa-final-recovery-20261023).

```bash
python run.py fetch-weights
python run.py verify-weights
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint final
```

Run from this model directory. Downloads verify every part and reconstructed file against its size and SHA-256. Identical frozen encoder bytes are reused locally.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `best/encoder.pt` | 2523197455 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--encoder.pt.part02) |
| `best/classifier.pt` | 18633 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--classifier.pt) |
| `final/encoder.pt` | 2523197455 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/model-weights-20261023/ijepa-vit-h14--encoder.pt.part02) |
| `final/classifier.pt` | 19807 | [download](https://github.com/RitwijParmar/tig-assessment/releases/download/ijepa-final-recovery-20261023/ijepa-vit-h14--final--classifier.pt) |

[manifest.json](manifest.json) records file hashes and exact pairings. Read the applicable [licenses](../licenses/). Reported benchmark metrics refer to **best**, not final.

## Final adaptation stage

`adaptation-final/` preserves the genuine epoch-5 JEPA target encoder and its original validation-selected classifier. It is separate from `final/`, which retains the last downstream head for the selected epoch-1 encoder. This stage is not the source of the reported 92.75% assessment result. No weights were retrained to recover this stage.

| File | Bytes | Downloads |
| --- | ---: | --- |
| `adaptation-final/encoder.pt` | 2523195875 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/ijepa-final-recovery-20261023/ijepa-vit-h14--adaptation-final--encoder.pt.part01), [part 2](https://github.com/RitwijParmar/tig-assessment/releases/download/ijepa-final-recovery-20261023/ijepa-vit-h14--adaptation-final--encoder.pt.part02) |
| `adaptation-final/classifier.pt` | 18633 | [part 1](https://github.com/RitwijParmar/tig-assessment/releases/download/ijepa-final-recovery-20261023/ijepa-vit-h14--adaptation-final--classifier.pt) |

[Stage descriptor](adaptation-final/model.json) records the exact pairing and original epoch identities.
