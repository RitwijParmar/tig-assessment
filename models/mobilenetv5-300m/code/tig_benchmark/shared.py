"""Reuse the published I-JEPA split, readout and metrics without changing it."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
IJEPA_ROOT = ROOT
sys.path.insert(0, str(ROOT / "code"))
from transfer_core.data import (
    normalize_rows,
    validate_roles,
    ResearchImages,
    TransformSpec,
)
from transfer_core.classifier import fit_probe
from transfer_core.metrics import quality_metrics, selection_score, class_weights
from transfer_core.utils import (
    atomic_json,
    atomic_torch,
    sha256,
    digest,
    configure_precision,
)

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
