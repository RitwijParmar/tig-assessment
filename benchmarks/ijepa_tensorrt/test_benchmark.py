"""CPU-only checks for the benchmark's measurement contract."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ijepa_benchmark", ROOT / "benchmark.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class NumericalGateTests(unittest.TestCase):
    def test_changed_decision_fails_even_with_small_probability_error(self):
        result = BENCHMARK.numerical_gate([-1e-6], [1e-6], 0.0001)
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision_disagreements"], 1)
        self.assertLess(result["max_abs_probability_difference"], 0.0001)

    def test_large_probability_error_fails_without_changed_decision(self):
        result = BENCHMARK.numerical_gate([1.0], [2.0], 0.0001)
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision_disagreements"], 0)

    def test_nonfinite_and_coverage_changes_are_rejected(self):
        self.assertFalse(BENCHMARK.numerical_gate([1.0], [np.nan], 0.0001)["passed"])
        with self.assertRaises(ValueError):
            BENCHMARK.numerical_gate([np.nan], [1.0], 0.0001)
        with self.assertRaises(ValueError):
            BENCHMARK.numerical_gate([1.0], [1.0, 1.0], 0.0001)

    def test_matching_logits_pass(self):
        result = BENCHMARK.numerical_gate([-1000.0, 0.0, 1000.0], [-1000.0, 0.0, 1000.0], 0.0001)
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision_disagreements"], 0)


class IsolationAndRecordedEvidenceTests(unittest.TestCase):
    def test_timing_samples_exclude_validation_and_assessment(self):
        roles = {role: [dict(group=f"{role}-group", label6=0, frame=i, path=f"{role}/{i}.png") for i in range(8)] for role in ("fitting", "selection", "assessment")}
        rows = BENCHMARK.balanced_rows(roles, 8, allowed=("fitting",))
        self.assertEqual({row["role"] for row in rows}, {"fitting"})
        self.assertEqual(len({row["path"] for row in rows}), 8)
        self.assertEqual(rows, BENCHMARK.balanced_rows(roles, 8, allowed=("fitting",)))
        development = BENCHMARK.balanced_rows(roles, 16)
        self.assertEqual({row["role"] for row in development}, {"fitting", "selection"})

    def test_recorded_ratios_and_full_agreement(self):
        result = json.loads((ROOT / "recorded_results.json").read_text())
        for boundary in result["comparison"].values():
            ratio = boundary["pytorch_fp32"]["median_ms"] / boundary["tensorrt_fp32"]["median_ms"]
            self.assertAlmostEqual(ratio, boundary["median_speedup"], places=12)
        gate = result["validation"]["tensorrt_gate"]
        self.assertEqual(result["validation"]["frames"], 6454)
        self.assertEqual(gate["frames"], 6454)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["decision_disagreements"], 0)
        self.assertFalse(result["method"]["assessment_images_loaded"])
        self.assertFalse(result["method"]["jetson_measured"])
        for excluded in result["excluded_variants"].values():
            self.assertFalse(excluded["development_gate"]["passed"])

    def test_packaged_source_identity_and_default_model_path(self):
        result = json.loads((ROOT / "recorded_results.json").read_text())
        self.assertEqual(hashlib.sha256((ROOT / "benchmark.py").read_bytes()).hexdigest(), result["provenance"]["packaged_worker_sha256"])
        self.assertEqual(BENCHMARK.EXECUTED_SOURCE_SHA256, result["provenance"]["executed_worker_sha256"])
        self.assertTrue((BENCHMARK.DEFAULT_MODEL_DIR / "config.json").is_file())

    def test_cli_help_does_not_require_cuda_imports(self):
        completed = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--help"], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--include-fp16", completed.stdout)
        self.assertIn("--manifest", completed.stdout)
        self.assertNotIn("\"stage\": \"failed\"", completed.stdout)
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
