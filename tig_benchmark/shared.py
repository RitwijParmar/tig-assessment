"""Reuse the published I-JEPA split, readout and metrics without changing it."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IJEPA_ROOT = ROOT / "models/ijepa-vit-h14"
if (
    "ijepa" in sys.modules
    and Path(sys.modules["ijepa"].__file__).resolve().parent != IJEPA_ROOT / "ijepa"
):
    raise RuntimeError("An unrelated ijepa package is already imported")
sys.path.insert(0, str(IJEPA_ROOT))
from ijepa.data import normalize_rows, validate_roles, ResearchImages, TransformSpec
from ijepa.classifier import fit_probe
from ijepa.metrics import quality_metrics, selection_score, class_weights
from ijepa.utils import atomic_json, atomic_torch, sha256, digest, configure_precision

__all__ = [
    "ROOT",
    "IJEPA_ROOT",
    "normalize_rows",
    "validate_roles",
    "ResearchImages",
    "TransformSpec",
    "fit_probe",
    "quality_metrics",
    "selection_score",
    "class_weights",
    "atomic_json",
    "atomic_torch",
    "sha256",
    "digest",
    "configure_precision",
]
