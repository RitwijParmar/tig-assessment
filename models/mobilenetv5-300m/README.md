# MobileNetV5-300M + linear classifier

## Status

Completed GPU training and retrospective held-out assessment on exactly the final adapted I-JEPA split. Selected mode: **supervised**. Best classifier epoch **16**; final classifier epoch **20**. [model.json](model.json), [config.json](config.json), [data.json](data.json) and [results.json](results.json) record the package, recipe, split and measured results. This folder contains its own implementation and tests.

## Architecture

Released Gemma 3n vision encoder, including its native multiscale fusion adapter and global pooling. Native 768px evaluation yields a 2048-dimensional image vector.

```text
Image → upstream pretrained encoder/readout → fitting-only feature normalization
      → Linear(2048, 1) → sigmoid → good / defect at threshold 0.5
```

The selected checkpoint updates the complete MobileNetV5 encoder and the linear classifier by gradients during supervised fitting. A frozen-encoder classifier was trained first and supplied the classifier initialization and epoch-0 baseline. Validation selected the final configuration.

## Data

Exactly the final adapted I-JEPA recording split: fitting **12 recordings / 10,338 frames**, validation **6 / 6,454**, assessment **9 / 6,763**. [data.json](data.json) includes class counts and exact normalized-row hashes. All recording segments remain together; no random frame split or fitting-plus-validation refit is used. These canonical recording roles supersede the original archive folder names; a path containing `train` or `test` is not itself a role assignment for this benchmark.

Dataset: [TIG Aluminium 5083 HDR camera dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`. Assessment recordings were inspected in earlier research, so these scores are retrospective held-out results.

## Training and selection

Use the shared data preparation, classifier recipe and evaluation commands. Raw paths, annotations, labels and checksums must match the [split fingerprints](data.json). Dataset: [TIG Aluminium 5083](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083); binary target `int(label6 > 0)`.

The same existing classifier-fitting function, fitting-only normalization, balanced BCE, optimizer, patience, selection metric and threshold are shared with I-JEPA. Model-native preprocessing and its full-frame geometry control are declared separately. Assessment is excluded from fitting, normalization and checkpoint choice, and requires a sealed verified run. Previously inspected recordings provide retrospective assessment rather than a new blind test.

The completed supervised configuration used encoder learning rate `1e-5`, head learning rate `1e-3`, effective batch 32 (physical batch8), 20 epochs and one warmup epoch. Before assessment, the user stopped the additional search: `3e-5` ended after five committed epochs and was excluded from final selection; `1e-4` was not started. This does not establish that either higher rate would perform worse. The frozen baseline and completed `1e-5` run were eligible for the final choice. The original three-rate recipe remains in config.json, and results.json records the amendment.

A fitting-only diagnostic found near-overflow stem gradients. Backward loss scaling by `2**-64` and float64 norm accumulation implement the same norm-1 global clipping without overflowing; assessment had not been opened.

## Results

| Metric | Validation: 6 recordings | Assessment: 9 recordings |
| --- | ---: | ---: |
| Recording-weighted balanced accuracy | 92.55% | 79.57% |
| Defect recall | 88.53% | 81.51% |
| False-alarm rate | 3.43% | 22.38% |
| AUROC | 0.9847 | 0.8683 |
| False-positive frames | 33 | 524 |
| False-negative frames | 390 | 594 |

Each frame has weight `1 / frames_in_its_recording`; class recalls are calculated with those weights and averaged for balanced accuracy. Raw frame counts have a different denominator. Full aggregate metrics and training/validation histories are in [results.json](results.json). Assessment was opened only after both new models were selected and reloaded successfully. No threshold tuning, fitting-plus-validation refit or assessment-driven retraining was performed.

## Usage

Run from this model directory:

```bash
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py test
python run.py --help
```

```bash
python run.py prepare --metadata /absolute/path/registration.json \
  --data-root /absolute/path/dataset-root --output runs/prepared
python run.py fetch-assets --output assets/pretrained
python run.py train --data runs/prepared/data.local.json --assets assets/pretrained/assets.local.json \
  --output runs/training --device cuda --batch-size 8 --workers 4 --finetune
python run.py verify --run runs/training --device cuda
python run.py evaluate --run runs/training --device cuda
```

The model identifier is supplied automatically by this folder’s runner. Asset setup verifies pinned upstream source and checkpoint bytes. The released encoder is available through the public weight downloader; the asset command additionally sets up pinned upstream source. Both selected checkpoints were sealed and replayed before either assessment was opened.


## Weights

`weights/best/` contains the validation-selected encoder and trained classifier. `weights/final/` contains the last completed epoch of that same selected configuration. Both include fitting-only normalization in the classifier checkpoint. The reported assessment uses **best**. A frozen encoder can have identical bytes in both folders. Best and final can also coincide when the last epoch is validation-best; both folders retain their defined checkpoints.

Use `python run.py fetch-weights` and `python run.py verify-weights`. [weights/README.md](weights/README.md) gives download links, and [weights/manifest.json](weights/manifest.json) records every file, size and SHA-256. Large encoder files are release downloads; the small best/final classifiers and portable descriptors are also committed in these folders. Both checkpoints reload on CPU and GPU and agree on the checked fitting-image decisions. CPU logits differ substantially, so the reported GPU metrics have not been established as CPU metrics.

```bash
python run.py fetch-assets --output assets/pretrained
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint best
python run.py predict-local --image /absolute/path/weld.png --device cpu --checkpoint final
```

## Implementation and licenses

Implementation lives in [code/](code/) and tests in [tests/](tests/). This model imports no code from another model folder. Read [licenses/](licenses/) for applicable source and checkpoint terms. Pinned upstream source is checked before loading. Checkpoints retain the exact saved training bytes; package/export code does not retrain or alter tensors. No classifier loss or EMA update is added to the released encoder during frozen-feature training.
