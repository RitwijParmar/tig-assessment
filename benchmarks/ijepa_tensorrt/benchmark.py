"""PyTorch/TensorRT comparison for the selected adapted I-JEPA ViT-H/14.

Portable packaging of the executed IEEE-FP32 benchmark. See README.md for
provenance, timing boundaries and the recorded results. Requires Linux/CUDA for
a new run. --help and the companion unit tests do not require PyTorch.
"""
from __future__ import annotations

import argparse
import collections
import gc
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import zlib

import numpy as np
from PIL import Image

ENCODER_SHA = "9f9c3dfe31e49c1a7333ac85e436d6a7b9f7a12c3a4b142ee12b662a7cde5b17"
HEAD_SHA = "990833bb02a531cb2350b65cb4141630b721d79eed62b9b5cc11b245f93619bb"
BUILD_POLICY = "ieee-fp32-mixed-fp16-obey-sensitive-v2"
EXECUTED_SOURCE_SHA256 = "e413fa6e5c472a0676267ebc4b62279c3e4b3968545bec8185ee693e6b147b1a"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models/ijepa-vit-h14-adapted"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def progress(stage, **details):
    print(json.dumps({"stage": stage, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}), flush=True)


def ieee_precision():
    # Do not call the packaged configure_precision(), which enables TF32.
    # The exporter itself uses legacy backend flag contexts. On Torch 2.14,
    # assigning only new precision properties leaves its separate legacy cuDNN
    # bool inconsistent, so use the legacy setters consistently in this process.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("The requested IEEE FP32 precision policy did not apply")


def balanced_rows(roles, count, allowed=("fitting", "selection")):
    """Fixed group/subtype coverage, independent of scores or mistakes."""
    cells = collections.defaultdict(list)
    for role in allowed:
        for row in roles[role]:
            cells[(role, row["group"], row["label6"])].append(row)
    ranked = []
    for key in sorted(cells):
        values = sorted(cells[key], key=lambda r: (r["frame"], r["path"]))
        idx = np.linspace(0, len(values) - 1, min(8, len(values))).round().astype(int)
        ranked.append([(key[0], values[i]) for i in idx])
    result, seen = [], set()
    for rank in range(max(map(len, ranked))):
        for cell in ranked:
            if rank < len(cell):
                role, row = cell[rank]
                if row["path"] not in seen:
                    result.append({"role": role, **row})
                    seen.add(row["path"])
                if len(result) == count:
                    return result
    raise ValueError("Not enough deterministic development examples")


def memory():
    free, total = torch.cuda.mem_get_info()
    result = {"device_used_bytes": int(total - free), "torch_allocated_bytes": int(torch.cuda.memory_allocated()), "torch_reserved_bytes": int(torch.cuda.memory_reserved()), "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        result["nvml_process_used_bytes"] = sum(int(p.usedGpuMemory) for p in processes if p.pid == os.getpid())
        result["power_watts"] = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000
        result["temperature_c"] = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception as error:
        result["nvml_unavailable"] = str(error)
    return result


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"samples": int(len(values)), "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)), "p99_ms": float(np.quantile(values, .99)), "mean_ms": float(values.mean()), "minimum_ms": float(values.min()), "maximum_ms": float(values.max()), "raw_ms": values.tolist()}


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def numerical_gate(reference, actual, probability_tolerance):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if not np.isfinite(reference).all():
        raise ValueError("The FP32 reference contains nonfinite logits")
    finite = bool(np.isfinite(actual).all())
    if reference.shape != actual.shape:
        raise ValueError("Parity output coverage changed")
    if not finite:
        return {"passed": False, "finite": False, "frames": len(reference), "probability_tolerance": probability_tolerance}
    delta = abs(sigmoid(reference) - sigmoid(actual))
    disagreements = int(((reference >= 0) != (actual >= 0)).sum())
    return {"passed": bool(disagreements == 0 and delta.max() <= probability_tolerance), "finite": True, "frames": len(reference), "max_abs_logit_difference": float(abs(reference - actual).max()), "max_abs_probability_difference": float(delta.max()), "mean_abs_probability_difference": float(delta.mean()), "decision_disagreements": disagreements, "probability_tolerance": probability_tolerance, "threshold": .5}


class TorchRunner:
    def __init__(self, model, amp=False):
        self.model, self.amp = model, amp

    def __call__(self, image, valid):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            return self.model(image, valid).float()


class TensorRTRunner:
    def __init__(self, engine_path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution context failed")
        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            if any(n < 0 for n in shape):
                raise ValueError("Only fixed batch-one engines are benchmarked")
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = (shape, dtype)
            else:
                self.outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
        if set(self.inputs) != {"images", "valid"} or set(self.outputs) != {"logit"}:
            raise ValueError("Unexpected TensorRT model inputs or outputs")
        for name, tensor in self.outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

    def __call__(self, image, valid):
        for name, tensor in (("images", image), ("valid", valid)):
            shape, dtype = self.inputs[name]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError("TensorRT input shape/dtype/contiguity changed")
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.outputs["logit"]


def build_engine(onnx_path, output, fp16):
    import tensorrt as trt
    if output.exists():
        metadata_path = output.with_suffix(".build.json")
        if not metadata_path.exists():
            raise ValueError("Incomplete engine build lacks provenance metadata")
        metadata = json.loads(metadata_path.read_text())
        if (metadata["engine_sha256"] != sha(output)
            or metadata["source_onnx_sha256"] != sha(onnx_path)
            or metadata["tensorrt"] != trt.__version__
            or metadata["compute_capability"] != list(torch.cuda.get_device_capability(0))
            or metadata["gpu"] != torch.cuda.get_device_name(0)
            or metadata.get("build_policy") != BUILD_POLICY
            or metadata["FP16"] != fp16):
            raise ValueError("Existing engine provenance, hardware or source changed")
        progress("engine_reuse", path=str(output), sha256=sha(output))
        return
    progress("engine_build_start", precision="FP16-mixed" if fp16 else "FP32-IEEE")
    start = time.perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # Passing a file path is essential for resolving >2GiB external weights.
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + json.dumps(errors))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.builder_optimization_level = 3
    config.max_aux_streams = 0
    constrained = []
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        sensitive = {trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX, trt.LayerType.REDUCE}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in sensitive:
                layer.precision = trt.float32
                if layer.type == trt.LayerType.NORMALIZATION and hasattr(layer, "compute_precision"):
                    layer.compute_precision = trt.float32
                for j in range(layer.num_outputs):
                    if layer.get_output(j).dtype in (trt.float32, trt.float16):
                        layer.set_output_type(j, trt.float32)
                constrained.append({"name": layer.name, "type": str(layer.type)})
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build returned no engine")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(output)
    save(output.with_suffix(".build.json"), {"seconds": time.perf_counter() - start, "tensorrt": trt.__version__, "TF32": False, "FP16": fp16, "build_policy": BUILD_POLICY, "fp32_sensitive_layers": constrained, "engine_sha256": sha(output), "engine_bytes": output.stat().st_size, "source_onnx_sha256": sha(onnx_path), "builder_worker_sha256": sha(__file__), "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0))})
    del serialized, parser, network, config, builder
    gc.collect()
    torch.cuda.empty_cache()
    progress("engine_build_complete", precision="FP16-mixed" if fp16 else "FP32-IEEE", seconds=time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Selected adapted I-JEPA model folder")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Prepared manifest.local.json with locked fitting/selection/assessment roles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--validation-limit", type=int, default=0, help="0 means all validation frames; a nonzero limit is diagnostic only")
    parser.add_argument("--resume", action="store_true", help="Reuse verified conversion artifacts; inference timings run again")
    parser.add_argument("--include-fp16", action="store_true", help="Also screen the historically rejected FP16 variants; no speedup is accepted unless both parity gates pass")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 20 or args.max_seconds < 60 or args.validation_limit < 0:
        raise ValueError("Invalid benchmark configuration")
    global torch
    import torch

    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a fresh output directory or explicit --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def expired(*_):
        raise TimeoutError("Worker wall-clock limit reached")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required; CPU timings are not substituted")
    # Keep this reference alive through main. Both runtimes, transfers and CUDA
    # events use this same nondefault stream for the entire benchmark.
    benchmark_stream = torch.cuda.Stream(device=torch.cuda.current_device())
    torch.cuda.set_stream(benchmark_stream)
    if torch.cuda.current_stream().cuda_stream == torch.cuda.default_stream().cuda_stream:
        raise RuntimeError("The benchmark requires a shared nondefault CUDA stream")
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.model_dir.resolve() / "code"))
    from ijepa.classifier import WeldClassifier
    from ijepa.data import content_mask, preprocessing, validate_roles
    from ijepa.metrics import quality_metrics
    inventory = json.loads((args.model_dir / "weights/manifest.json").read_text())
    if inventory["model_id"] != "ijepa-vit-h14-adapted":
        raise ValueError("Expected the selected adapted I-JEPA model package")
    encoder = args.model_dir / "weights" / inventory["artifacts"]["best_encoder"]["path"]
    head = args.model_dir / "weights" / inventory["artifacts"]["best_classifier"]["path"]
    if sha(encoder) != ENCODER_SHA or sha(head) != HEAD_SHA:
        raise ValueError("Selected best checkpoint hashes changed")
    config = json.loads((args.model_dir / "config.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    roles = manifest["roles"]
    validate_roles(roles, json.loads((args.model_dir / "data.json").read_text()))
    transform = preprocessing(config)
    parity_rows = balanced_rows(roles, 64)
    timing_rows = balanced_rows(roles, 8, allowed=("fitting",))
    plan = {"model_id": inventory["model_id"], "encoder_sha256": ENCODER_SHA, "classifier_sha256": HEAD_SHA, "manifest_sha256": sha(args.manifest), "fixed_batch": 1, "image_shape": [1, 3, 224, 224], "valid_shape": [1, 256], "valid_dtype": "float32 zeros/ones; mathematically identical valid-patch pooling", "warmup": args.warmup, "iterations": args.iterations, "validation_limit": args.validation_limit, "assessment_accessed": False, "numerical_gate": {"FP32_max_probability_error": .0001, "FP16_max_probability_error": .005, "maximum_decision_disagreements": 0}, "precision_policy": "TF32 disabled in PyTorch and TensorRT. FP16 Torch uses autocast; TensorRT uses mixed FP16 with FP32 normalization, softmax and reductions.", "timing_policy": "Serial batch1, no CUDA graphs, one shared nondefault CUDA stream for PyTorch and TensorRT, no concurrent streams. GPU CUDA-event compute and synchronized host-logit wall timings separated. PNG-bytes-to-decision includes decode, exact preprocessing, transfer, network, sigmoid and returned decision; disk read and model loading excluded.", "parity_rows": parity_rows, "timing_rows": timing_rows, "include_fp16": args.include_fp16, "executed_source_sha256": EXECUTED_SOURCE_SHA256, "inference_only": True, "original_checkpoints_read_only": True, "not_a_jetson_measurement": True}
    if (args.output / "Plan.json").exists():
        old = json.loads((args.output / "Plan.json").read_text())
        if old != plan:
            raise ValueError("Resume plan or input identities changed")
    save(args.output / "Plan.json", plan)
    import tensorrt as trt
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory, "worker_sha256": sha(__file__), "model_source_sha256": {str(p.relative_to(args.model_dir)): sha(p) for p in sorted((args.model_dir / "code").rglob("*.py"))}, "driver_csv": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,pstate,power.limit", "--format=csv,noheader"], text=True).strip()}
    save(args.output / ("Environment_" + str(time.time_ns()) + ".json"), environment)
    save(args.output / "Environment.json", environment)
    ieee_precision()

    def decode(row, raw=None, verify=True):
        raw = (args.data_root / row["path"]).read_bytes() if raw is None else raw
        if verify and (len(raw) != row["original_bytes"] or zlib.crc32(raw) & 0xffffffff != row["crc"]):
            raise ValueError("Dataset image bytes changed: " + row["path"])
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            valid = content_mask(image.size).float()[None]
            # PIL preprocessing yields a strided CHW view. Use the same dense
            # NCHW layout in both runtimes, with the copy inside PNG timing.
            tensor = transform(image)[None].contiguous()
        return tensor, valid

    parity_inputs = [decode(row) for row in parity_rows]
    timing_inputs = [decode(row) for row in timing_rows]
    timing_bytes = [(args.data_root / row["path"]).read_bytes() for row in timing_rows]
    validation_rows = roles["selection"][:args.validation_limit or None]
    result = {"scope": "Selected adapted I-JEPA only. Fixed weights, no model fitting or assessment-driven selection.", "variants": {}}
    fp32_reference = None
    validation_reference = None

    def run_variant(name, make_runner, tolerance):
        nonlocal fp32_reference, validation_reference
        progress("variant_start", variant=name)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = memory()
        runner = make_runner()
        torch.cuda.synchronize()
        samples = [memory()]
        actual = []
        for image, valid in parity_inputs:
            actual.append(float(runner(image.cuda(), valid.cuda())[0].item()))
        if fp32_reference is None:
            fp32_reference = actual
        gate = numerical_gate(fp32_reference, actual, tolerance)
        detail = {"development_gate": gate, "parity_logits": [v if np.isfinite(v) else None for v in actual], "latency_result_eligible": gate["passed"], "memory_before_load": before}
        if not gate["passed"]:
            detail["status"] = "numerical_gate_failed; no speedup claimed"
            result["variants"][name] = detail
            save(args.output / "Results.json", result)
            del runner
            gc.collect()
            torch.cuda.empty_cache()
            progress("variant_rejected", variant=name, gate=gate)
            if name == "pytorch_fp32":
                raise RuntimeError("The FP32 reference failed its own finiteness gate")
            return
        gpu_inputs = [(x.cuda(), v.cuda()) for x, v in timing_inputs]
        for i in range(args.warmup):
            image, valid = gpu_inputs[i % len(gpu_inputs)]
            runner(image, valid)
        torch.cuda.synchronize()
        samples.append(memory())
        device_ms, wall_ms, decoded_ms = [], [], []
        for i in range(args.iterations):
            image, valid = gpu_inputs[i % len(gpu_inputs)]
            torch.cuda.synchronize()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            start_event.record()
            value = runner(image, valid)
            end_event.record()
            _ = float(value[0].item())
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - start) * 1000)
            device_ms.append(start_event.elapsed_time(end_event))
        # Separate end-to-end boundary on the exact same in-memory PNG bytes.
        for i in range(args.warmup):
            j = i % len(timing_rows)
            image, valid = decode(timing_rows[j], timing_bytes[j], verify=False)
            probability = torch.sigmoid(runner(image.cuda(), valid.cuda()).float())[0].item()
            _ = bool(probability >= .5)
        torch.cuda.synchronize()
        for i in range(args.iterations):
            j = i % len(timing_rows)
            torch.cuda.synchronize()
            start = time.perf_counter()
            image, valid = decode(timing_rows[j], timing_bytes[j], verify=False)
            probability = torch.sigmoid(runner(image.cuda(), valid.cuda()).float())[0].item()
            _ = bool(probability >= .5)
            torch.cuda.synchronize()
            decoded_ms.append((time.perf_counter() - start) * 1000)
        samples.append(memory())
        logits = []
        for i, row in enumerate(validation_rows):
            image, valid = decode(row)
            logits.append(float(runner(image.cuda(), valid.cuda())[0].item()))
            if (i + 1) % 500 == 0:
                progress("validation_progress", variant=name, frames=i + 1, total=len(validation_rows))
        if validation_reference is None:
            validation_reference = logits
        full_gate = numerical_gate(validation_reference, logits, tolerance)
        if full_gate["finite"]:
            predictions, metrics = quality_metrics(validation_rows, torch.tensor(logits, dtype=torch.float32))
            save(args.output / (name + "_validation_predictions.json"), predictions)
        else:
            metrics = None
            save(args.output / (name + "_validation_logits_failed.json"), [{"path": row["path"], "logit": v if np.isfinite(v) else None} for row, v in zip(validation_rows, logits)])
        detail.update({"status": "completed" if full_gate["passed"] else "validation_parity_failed", "gpu_compute": stats(device_ms), "input_tensor_to_host_logit": stats(wall_ms), "png_bytes_to_host_decision": stats(decoded_ms), "memory_samples": samples + [memory()], "memory_scope": "Sampled runtime process/device memory after load, warmup and inference. TensorRT memory is outside Torch allocator counters. Engine build memory excluded. Device count may include unrelated processes.", "validation_gate": full_gate, "validation_metrics": metrics, "validation_complete": len(validation_rows) == len(roles["selection"]), "validation_decisions_identical_to_fp32": full_gate.get("decision_disagreements") == 0, "latency_result_eligible": bool(gate["passed"] and full_gate["passed"] and len(validation_rows) == len(roles["selection"]))})
        result["variants"][name] = detail
        save(args.output / "Results.json", result)
        progress("variant_complete", variant=name, development_gate=gate, validation_gate=full_gate, gpu_median_ms=detail["gpu_compute"]["median_ms"], png_median_ms=detail["png_bytes_to_host_decision"]["median_ms"])
        del gpu_inputs, runner
        gc.collect()
        torch.cuda.empty_cache()

    # The same CPU module moves to CUDA for each Torch phase, then back to CPU.
    model = WeldClassifier.from_files(encoder, head, "cpu", expected={"encoder": ENCODER_SHA, "classifier": HEAD_SHA})
    def torch_runner(amp):
        model.to("cuda").eval()
        ieee_precision()
        return TorchRunner(model, amp)
    run_variant("pytorch_fp32", lambda: torch_runner(False), .0001)
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    if args.include_fp16:
        run_variant("pytorch_autocast_fp16", lambda: torch_runner(True), .005)
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    onnx_path = args.output / "ijepa_best.onnx"
    if onnx_path.exists():
        export_metadata = args.output / "ONNX_Export.json"
        if not export_metadata.exists():
            raise ValueError("An incomplete ONNX export exists without provenance metadata")
        for item in json.loads(export_metadata.read_text())["files"]:
            if sha(args.output / item["name"]) != item["sha256"]:
                raise ValueError("Existing ONNX graph or external weights changed")
    if not onnx_path.exists():
        progress("onnx_export_start")
        start = time.perf_counter()
        torch.onnx.export(model, parity_inputs[0], str(onnx_path), input_names=["images", "valid"], output_names=["logit"], opset_version=18, dynamo=True, external_data=True, optimize=False)
        import onnx
        onnx.checker.check_model(str(onnx_path))
        save(args.output / "ONNX_Export.json", {"passed": True, "seconds": time.perf_counter() - start, "external_data": True, "opset": 18, "worker_sha256": sha(__file__), "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(args.output.glob("ijepa_best.onnx*"))]})
        progress("onnx_export_complete", seconds=time.perf_counter() - start)
    # Independent CPU ONNX check before TensorRT, using three development frames.
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    indices = [0, len(parity_inputs) // 2, len(parity_inputs) - 1]
    ort_logits = [float(session.run(["logit"], {"images": parity_inputs[i][0].numpy(), "valid": parity_inputs[i][1].numpy()})[0][0]) for i in indices]
    ort_gate = numerical_gate([fp32_reference[i] for i in indices], ort_logits, .0001)
    save(args.output / "ONNX_Parity.json", {"indices": indices, "logits": ort_logits, "comparison": "CPU ONNX Runtime versus CUDA IEEE PyTorch FP32", **ort_gate})
    del session, model
    gc.collect()
    if not ort_gate["passed"]:
        raise RuntimeError("ONNX numerical parity failed; no TensorRT timing is accepted")
    trt_variants = [("tensorrt_fp32", False, .0001)]
    if args.include_fp16:
        trt_variants.append(("tensorrt_fp16", True, .005))
    for name, half, tolerance in trt_variants:
        engine = args.output / (name + ".engine")
        build_engine(onnx_path, engine, half)
        run_variant(name, lambda p=engine: TensorRTRunner(p), tolerance)
    hashes = {"encoder": sha(encoder), "classifier": sha(head), "manifest": sha(args.manifest)}
    assert hashes == {"encoder": ENCODER_SHA, "classifier": HEAD_SHA, "manifest": plan["manifest_sha256"]}
    result["preservation_after"] = hashes
    result["complete"] = True
    result["not_a_jetson_measurement"] = True
    save(args.output / "Results.json", result)
    signal.alarm(0)
    progress("complete", variants=list(result["variants"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        progress("failed", error=repr(error), traceback=traceback.format_exc())
        raise
