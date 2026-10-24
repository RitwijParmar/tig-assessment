# TIG Assessment

Offline good-versus-defective TIG weld classification with four models on the same recording splits. The selected weld-adapted **I-JEPA ViT-H/14 + linear classifier** achieved **92.75% balanced accuracy** and **96.51% defect recall** on assessment. Its main limitation is a concentrated false-alarm failure in one recording.

[One-page report](docs/report.pdf) | [Findings and follow-up](docs/findings.md) | [PyTorch vs TensorRT code](benchmarks/ijepa_tensorrt/)

## Models and results

All values below are recording-weighted percentages at threshold **0.5**. BA means balanced accuracy, recall means defect recall, and FAR means false-alarm rate.

| Model | Validation BA | Recall | FAR | Assessment BA | Recall | FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [MobileNetV5-300M](models/mobilenetv5-300m/) | 92.55 | 88.53 | 3.43 | 79.57 | 81.51 | 22.38 |
| [DINOv3 ViT-L/16](models/dinov3-vit-l16/) | 89.70 | 85.91 | 6.51 | 88.00 | 83.45 | 7.45 |
| [I-JEPA ViT-H/14, pretrained](models/ijepa-vit-h14-pretrained/) | 94.78 | 89.72 | 0.16 | 84.89 | 85.02 | 15.25 |
| **[I-JEPA ViT-H/14, weld-adapted](models/ijepa-vit-h14-adapted/)** | **98.41** | **97.58** | **0.77** | **92.75** | **96.51** | **11.00** |

The adapted model had the highest observed assessment BA and defect recall. This compares the implemented pipelines on matched data. Their backbones, preprocessing and training recipes differ, so the scores do not isolate the effect of a pretraining objective or establish a statistically significant architecture advantage.

## Data and evaluation

Dataset: [TIG Aluminium 5083 HDR camera images](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083). Released labels 1-5 are grouped as defective, and label 0 is good.

| Split | Recordings | Frames | Use |
| --- | ---: | ---: | --- |
| Training / fitting | 12 | 10,338 | Adaptation, classifier fitting and normalization |
| Validation / selection | 6 | 6,454 | Checkpoint selection |
| Assessment | 9 | 6,763 | Evaluation of the selected checkpoint |

All four use identical frame identities and labels. Segments of the same recording stay together. Each model's `data.json` records the split and fingerprints. Frames receive weight `1 / number_of_frames_in_their_recording`; class recalls are calculated with these weights and averaged for BA. Raw frame counts have different denominators.

Assessment is excluded from this run's fitting, normalization and checkpoint selection. These recordings were inspected in earlier research, so the reported assessment is **retrospective held-out evidence**, rather than a new blind test.

## Why these approaches

I started with MobileNet because I had used it in my deep learning course. I revisited V5's convolution, attention and multiscale features and wanted to test its inference-oriented design. Results were decent but varied across recordings. DINOv3 gave me a self-supervised transfer benchmark. It reduced assessment false alarms relative to MobileNet, but still missed defects, particularly lack of penetration. Lowering its threshold traded more detections for more false alarms.

I had read about I-JEPA but had never implemented it. Predicting hidden-region embeddings seemed promising for weld features. Rechecking the first AI-generated implementation showed that it was supervised transfer from I-JEPA weights with LoRA and a classifier. The intended JEPA adaptation was missing. The corrected training path is:

```text
Same image
  visible patches -> context encoder -> predictor -> predicted target embeddings
  full crop       -> target encoder              -> target embeddings
                                      compare corresponding target patches

Adaptation: context encoder + predictor learn through gradients
            target encoder follows the context encoder through EMA
Readout:    freeze the adapted target encoder, then train a linear classifier
Inference:  one encoder + fitting-only normalization + saved linear classifier
```

The pretrained I-JEPA baseline uses its original frozen encoder. DINOv3 also uses a frozen encoder. The selected MobileNetV5 model uses supervised encoder and classifier fine-tuning. Each model folder documents its exact architecture and recipe.

## False alarms and follow-up

Adapted I-JEPA's FAR increased from **0.77% on validation to 11.00% on assessment**. Recording **`170906-141809` accounts for 514 of 521 false alarms**. The other eight recordings contribute seven, with a **0.48% recording-weighted FAR across their available good frames**. Only three of those eight contain good frames; five contain defects only. All nine remain in the reported results. The difficult recording also accounts for most of DINOv3's false alarms, while MobileNetV5's are more distributed.

Per-recording counts located the failure first. SliceFinder then identified associations with bright-region position and contrast within that recording. The same position marker appears in correctly classified good frames elsewhere, so it is an appearance pattern to investigate, not a proven physical cause or label error.

Three **15-epoch** frozen-encoder classifier trials tested continuation, targeted augmentation and augmentation with difficult-good-frame weighting. A separate nonlinear-classifier screen was also completed. None met the development requirement of reducing false alarms while preserving defect recall. The original model and published results were retained. See [findings](docs/findings.md) for the controls and limits.

## GPU inference

Only the selected adapted I-JEPA was benchmarked, on an **NVIDIA RTX PRO 6000 Blackwell Server Edition**. Both runtimes used matched IEEE FP32 with TF32 disabled, batch one, 30 warmups and 200 timed samples.

| Median timing boundary | PyTorch FP32 | TensorRT FP32 | Speedup |
| --- | ---: | ---: | ---: |
| Preprocessed GPU input to returned logit | 14.44 ms | 8.17 ms | 1.77x |
| In-memory PNG bytes to returned decision | 22.24 ms | 15.96 ms | 1.39x |

TensorRT matched all **6,454 validation decisions** against the fresh matched PyTorch reference. The PNG boundary includes decoding and preprocessing but excludes camera acquisition, disk reads and model loading. These are server results. Jetson performance has not been measured. FP16 variants failed the numerical acceptance gate and are not used for a speedup claim.

The [benchmark folder](benchmarks/ijepa_tensorrt/) contains the export/build/comparison script, dependencies, recorded summaries and reproduction instructions. Historical classification scores above remain unchanged.

## Run a model

Each model folder includes its training, evaluation, inference, tests and best-weight metadata. For the selected model:

```bash
git clone https://github.com/RitwijParmar/tig-assessment.git
cd tig-assessment/models/ijepa-vit-h14-adapted
python -m pip install -r requirements.txt
python run.py fetch-weights
python run.py verify-weights
python run.py fetch-source
python run.py predict --image /absolute/path/weld.png --device cpu
```

Weights are managed inside each model's `weights/` folder. Only the validation-selected best encoder and matching classifier are distributed. Large encoders are downloaded from the URLs in that model's manifest. I-JEPA's two encoder parts are pieces of one approximately 2.52 GB file; `fetch-weights` joins and verifies them automatically. Keep the encoder and classifier from the same model together.

For training or evaluation, follow the selected [model README](models/ijepa-vit-h14-adapted/README.md#training-and-evaluation), or the corresponding README in another model folder. `prepare` verifies the registered split. `train` fits the model, `evaluate` assesses a sealed run, and `evaluate-best` assesses the downloaded best pair without refitting. Reproducing JEPA training requires the official full pretraining checkpoint with both encoders and the predictor, as documented there.

From the repository root, `python verify.py` checks package structure, sources, split consistency, manifests and documentation links. Add `--weights` after downloading all four encoders to verify their bytes. Model tests run through `python run.py test` inside each model folder.

## Repository layout

```text
models/<model>/
  README.md, config.json, data.json, results.json
  run.py, requirements.txt, code/, tests/, licenses/
  weights/README.md, manifest.json, best/
benchmarks/ijepa_tensorrt/   # PyTorch/TensorRT comparison and recorded evidence
docs/report.pdf            # one-page project report
docs/findings.md           # investigation and follow-up details
docs/results.json          # aggregate results
verify.py
```

Model folders are self-contained. Shared utilities and required license notices are intentionally retained in each package. Dataset images, private frame-level outputs, generated ONNX files and hardware-specific TensorRT engines are not committed.

## Future direction

With control over a real welding setup, I would prioritize independent recordings under conditions similar to the difficult case, including good and defective welds. A data-coverage gap is my first suspicion, not an established cause. I would test better coverage before changing the model again.

Model work would remain my main focus. I would also improve the full pipeline as I did with TensorRT. Actual Jetson testing would focus mainly on **latency**, then memory use. In ROS 2 I would test image transport and QoS tuning, including a measured Fast DDS versus Connext comparison. Delivered FPS, latency and dropped frames would determine whether it helps. No DDS improvement percentage is assumed.
