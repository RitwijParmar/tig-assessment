# Findings and evaluation

This is an offline TIG weld-image study. The selected weld-adapted I-JEPA pipeline had the highest balanced accuracy and defect recall in the matched assessment below. That is an observed result on this split, not a statistically established advantage on new welding setups.

The [one-page report](report.pdf) gives the first-person research narrative. [results.json](results.json) contains the aggregate values used here.

## Data and selection

All four models use identical frame identities and labels from the TIG Aluminium 5083 dataset. Complete parent recordings stay in one split, including folders whose names end in `-partN`.

| Role | Recordings | Frames |
| --- | ---: | ---: |
| Training | 12 | 10,338 |
| Validation | 6 | 6,454 |
| Assessment | 9 | 6,763 |

Validation selects the reported checkpoint. Assessment is excluded from fitting, normalization and checkpoint selection. Those recordings had already been inspected in earlier research, so the assessment is retrospective held-out evidence rather than a new blind final test.

## Matched model results

All values are percentages at threshold 0.5. BA means balanced accuracy, recall means defect recall and FAR means false-alarm rate.

| Model | Validation BA | Recall | FAR | Assessment BA | Recall | FAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV5-300M | 92.55 | 88.53 | 3.43 | 79.57 | 81.51 | 22.38 |
| DINOv3 ViT-L/16 | 89.70 | 85.91 | 6.51 | 88.00 | 83.45 | 7.45 |
| I-JEPA ViT-H/14, pretrained | 94.78 | 89.72 | 0.16 | 84.89 | 85.02 | 15.25 |
| **I-JEPA ViT-H/14, weld-adapted** | **98.41** | **97.58** | **0.77** | **92.75** | **96.51** | **11.00** |

These are recording-weighted metrics. Each frame receives weight `1 / number of assessed frames in its parent recording`. Class recalls are computed with those weights, then averaged for BA. FAR uses the weighted good-frame denominator. It is not the pooled raw-frame error rate or a simple average of per-recording FARs. The source values remain in each model's `results.json`.

The split is matched, but backbone, preprocessing, readout and training recipe differ between pipelines. This comparison does not isolate the effect of the pretraining objective alone.

## I-JEPA training approach

The reported comparison evaluates the original pretrained representation and a weld-adapted representation. JEPA adaptation sends visible patches through a context encoder and predictor, then compares predicted embeddings with corresponding target-patch embeddings from the full-image target encoder. Gradients update the context encoder and predictor. An exponential moving average updates the target encoder.

After weld adaptation, the encoder is frozen and a separate linear classifier is trained. Inference uses that encoder and classifier. The original-pretrained baseline also uses a frozen encoder and its own trained linear classifier. Model folders retain their training, evaluation and inference code and the matching best weights.

## What the assessment showed

MobileNet's assessment false alarms were spread across several recordings. DINOv3 had fewer assessment false alarms, but lack of penetration accounted for 419 of its 658 missed defect frames. DINO detected 281 of 700 lack-of-penetration frames in this split, compared with 611 for adapted I-JEPA. Later diagnostics found interactions between DINO's CLS and patch-mean contributions and sensitivity to image framing. These retrospective diagnostics characterize the observed behavior without establishing a single architectural or physical cause.

For adapted I-JEPA, recording `170906-141809` contributed **514 of 521 assessment false alarms**. It contains 1,037 good frames, giving a raw within-recording FAR of 49.57%.

| Model | False alarms in this recording | All assessment false alarms |
| --- | ---: | ---: |
| Pretrained I-JEPA | 723 | 729 |
| Adapted I-JEPA | 514 | 521 |
| MobileNetV5 | 113 | 524 |
| DINOv3 | 269 | 310 |

The other eight recordings contributed only seven adapted-I-JEPA false alarms. Their recording-weighted FAR is **0.48%**, compared with 11.00% across the full assessment. Only three of those eight contain good frames. The five defect-only recordings cannot provide a FAR denominator. The full assessment score still includes the difficult recording.

## What SliceFinder added

Per-recording counts identified the concentration before slice discovery. SliceFinder then helped identify an appearance association within that recording. The search used 24 predefined visual measurements with bins derived from fitting images, rather than recording IDs as search features.

The strongest reported marker was a bright-region horizontal centroid below 0.489945 of image width. There were 509 false alarms among 747 good frames with that marker, versus 5 among 290 without it. The corresponding raw rates were 68.14% and 1.72%. An alternating-temporal-block check retained the association.

This is diagnostic evidence, not an independent-frame significance test or a causal rule. The same marker appeared in 311 correctly classified good frames from another assessment recording and in fitting good frames. Position alone is therefore insufficient to explain the observed errors. Contrast also varied with the error regime. Neither a physical defect cause nor incorrect labels was established. False alarms began well before the released label transition, so a small annotation-boundary shift cannot explain all errors.

## Follow-up experiments

Three classifier arms completed 15 epochs each with the encoder frozen: continuation control, targeted appearance augmentation, and augmentation with extra weight on difficult fitting good frames. Epoch zero remained eligible. No trained candidate reduced development false alarms while preserving the required defect recall and worst-good-recording behavior. The original classifier was retained. Rejected heads were not tested on assessment.

A separate RBF-SVM screen tested six settings over three whole-recording folds inside the original fitting set. None passed its false-alarm acceptance gate, so none advanced to validation or assessment. These were downstream-head screens conditional on the existing adapted encoder, not fully isolated end-to-end cross-validation. Neither study established a reduction for `170906-141809`.

The follow-up used a fresh IEEE FP32 reference. Small differences from historical TF32 execution were documented separately. The published model results above and original checkpoints were not replaced.

## Inference and future direction

The selected adapted I-JEPA was benchmarked on an NVIDIA RTX PRO 6000 Blackwell Server Edition. TensorRT FP32 reduced median model inference from 14.44 ms to 8.17 ms against matched PyTorch FP32, a 1.77x speedup. Including in-memory PNG decoding and preprocessing, latency fell from 22.24 ms to 15.96 ms, a 1.39x speedup. All 6,454 validation decisions agreed with the fresh matched PyTorch reference. Camera acquisition, disk reads and startup were outside that timing boundary. See the [benchmark code and recorded protocol](../benchmarks/ijepa_tensorrt/).

In a real welding setup, collecting more independent recordings resembling that case would be a sensible priority, including both good and defective welds. The concentration of errors makes recording coverage worth investigating, but does not prove that changing the data alone will resolve it. A welding expert could also review the appearance and label transition.

Jetson testing remains future work, especially latency and memory use. The server speedup does not predict a Jetson speedup. ROS 2 image transport, QoS and Fast DDS versus Connext would also need measurements of delivered FPS, end-to-end latency and dropped frames. No Jetson or DDS benchmark has been completed for this project.
