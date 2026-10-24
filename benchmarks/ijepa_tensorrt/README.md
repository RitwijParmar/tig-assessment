# PyTorch and TensorRT inference

This folder contains the export, engine build and comparison script for the selected weld-adapted I-JEPA ViT-H/14. It measures the saved encoder, fitting-derived feature normalization and linear classifier together. It does not train or change the model.

## Recorded result

Measured on an NVIDIA RTX PRO 6000 Blackwell Server Edition with batch one, IEEE FP32 and TF32 disabled in both runtimes.

| Timing boundary | PyTorch median / p95 | TensorRT median / p95 | Median speedup |
| --- | ---: | ---: | ---: |
| Preprocessed GPU input to returned logit | 14.44 / 14.45 ms | 8.17 / 8.19 ms | 1.77x |
| In-memory PNG bytes to returned decision | 22.24 / 22.80 ms | 15.96 / 16.56 ms | 1.39x |

The PNG boundary includes image decoding, letterboxing, the valid-patch mask, normalization, CPU/GPU transfer, the encoder, classifier, sigmoid and returned decision. It excludes camera acquisition, disk reads and model startup. It is not complete camera-to-decision latency.

TensorRT FP32 produced identical decisions on all 6,454 validation frames from six recordings against the fresh matched PyTorch reference. Maximum probability error was 0.00003520, below the predeclared 0.0001 tolerance. The fresh IEEE reference changes one validation-frame decision relative to the historical TF32 evaluation. The published model results are preserved.

Both FP16 variants failed the predeclared 0.005 probability-error gate on 64 development frames. Their unchanged decisions in that small sample did not qualify them for a speedup claim. The default command below therefore compares the two qualified FP32 paths. `--include-fp16` additionally reproduces the historical four-variant screening order.

[recorded_results.json](recorded_results.json) contains the measured timings, agreement checks, software versions and checkpoint hashes. These are saved results from 23 October 2026. Packaging and CPU checks of this folder did not rerun the GPU benchmark.

## Run the comparison

Use Linux with an NVIDIA CUDA GPU. The recorded software stack was Python 3.12.3, PyTorch 2.14.1+cu130, torchvision 0.29.1, CUDA 13.0 and TensorRT 10.13.3.9.post1. The requirements pin those recorded core versions. Additional ONNX/export tooling was not version-pinned in the original run.

From the repository root, install the matching CUDA PyTorch build in a separate environment, then the benchmark dependencies:

```bash
python -m venv .venv-benchmark
source .venv-benchmark/bin/activate
python -m pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r benchmarks/ijepa_tensorrt/requirements.txt
python models/ijepa-vit-h14-adapted/run.py fetch-weights
python models/ijepa-vit-h14-adapted/run.py verify-weights
```

Prepare the canonical dataset using the adapted model's [data preparation instructions](../../models/ijepa-vit-h14-adapted/README.md). Reuse the resulting `manifest.local.json`. The benchmark verifies its exact recording/frame fingerprints against the model package, then reads images from fitting and validation roles only. It reads assessment metadata to check split identity but does not load assessment images.

```bash
python benchmarks/ijepa_tensorrt/benchmark.py \
  --data-root /absolute/path/to/dataset-root \
  --manifest /absolute/path/to/prepared/manifest.local.json \
  --output /absolute/path/to/new-benchmark-run
```

`--model-dir` defaults to `models/ijepa-vit-h14-adapted` in this repository. The selected encoder and classifier SHA-256 values are checked before and after the run. A new output directory is required. `--resume` reuses only conversion artifacts whose source/runtime/hardware metadata and hashes match. Inference, agreement checks and timings run again.

The script first measures PyTorch FP32, exports an opset-18 ONNX model with external weights, checks three development examples in CPU ONNX Runtime, builds a TensorRT FP32 engine, then measures it. Both runtimes must pass the same probability and decision gates on 64 development examples and the full validation set before a latency result is eligible. Each timing boundary uses 30 warmups and 200 serial measurements over eight deterministic fitting images. Both runtimes use one shared nondefault CUDA stream with CUDA graphs disabled.

Generated artifacts include the ONNX graph and external weights, target-specific TensorRT engine, build metadata, local prediction files and a new `Results.json`. Keep these local artifacts outside Git. Do not publish per-frame outputs or replace the recorded results with a new run without identifying its environment.

`--validation-limit N` is a diagnostic shortcut. A limited check does not establish agreement over the full validation set. No speedup from such a run should be described as reproducing the complete validation check. The default is all 6,454 frames.

## Source and checks

The packaged script is derived from the worker that actually produced the recorded result. Its original SHA-256 is `e413fa6e5c472a0676267ebc4b62279c3e4b3968545bec8185ee693e6b147b1a`. The original timing functions, ONNX settings, TensorRT build policy and numerical gates are retained. Packaging adds a portable model path, clearer CLI help, delayed PyTorch import and an optional FP16 screen. It also prevents a limited-validation diagnostic from being marked eligible for a full-validation latency claim. The original and packaged hashes are recorded in the JSON summary.

CPU checks need only NumPy and Pillow, not PyTorch, TensorRT or a GPU:

```bash
python benchmarks/ijepa_tensorrt/benchmark.py --help
python -m unittest discover -s benchmarks/ijepa_tensorrt -p 'test_*.py' -v
```

These check sample-role isolation, parity-gate behavior, recorded ratios and packaged-source identity. They do not verify a new ONNX export or TensorRT execution. Those require the CUDA run above.

Only the selected adapted I-JEPA was benchmarked. The recorded run used one fixed runtime order and is not a thermal stress test. Server speedups do not predict a Jetson speedup ratio. Build a new engine on the intended target and measure latency and memory use there. ROS 2/DDS performance has not been measured in this project.
