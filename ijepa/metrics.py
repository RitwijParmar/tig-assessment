"""Explicit parent-weighted classification and calibration metrics."""

from __future__ import annotations

from collections import Counter
import numpy as np

from .data import checked_rows

LABELS = (
    "good",
    "burn_through",
    "contamination",
    "lack_of_fusion",
    "misalignment",
    "lack_of_penetration",
)


def _arrays(rows, score_key="score_raw", threshold=0.5):
    rows = checked_rows(rows)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be predeclared in [0,1]")
    scores = np.asarray([float(r[score_key]) for r in rows], dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities")
    y = np.asarray([r["target"] for r in rows], dtype=np.int64)
    counts = Counter(r["group"] for r in rows)
    weights = np.asarray([1 / counts[r["group"]] for r in rows])
    return rows, y, scores, weights


def weighted_auc(y, scores, weights):
    """Weighted Mann-Whitney AUROC, with exact half credit for ties."""
    y, scores, weights = (
        np.asarray(y),
        np.asarray(scores),
        np.asarray(weights, dtype=float),
    )
    if (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError("Invalid AUROC weights")
    positive, negative = weights[y == 1].sum(), weights[y == 0].sum()
    if positive <= 0 or negative <= 0:
        return None
    order = np.argsort(scores, kind="stable")
    s, targets, w = scores[order], y[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    pw = np.add.reduceat(w * (targets == 1), starts)
    nw = np.add.reduceat(w * (targets == 0), starts)
    return float((pw * (np.cumsum(nw) - nw + 0.5 * nw)).sum() / (positive * negative))


def _values(y, scores, weights, threshold=0.5, bins=10):
    pred = scores >= threshold
    positive, negative = weights[y == 1].sum(), weights[y == 0].sum()
    good = float(weights[(y == 0) & ~pred].sum() / negative) if negative > 0 else None
    defect = float(weights[(y == 1) & pred].sum() / positive) if positive > 0 else None
    total = weights.sum()
    if total <= 0:
        return {
            key: None
            for key in (
                "balanced_accuracy",
                "good_recall",
                "defect_recall",
                "auroc",
                "brier",
                "ece",
            )
        }
    bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
    ece, table = 0.0, []
    for k in range(bins):
        keep = bin_ids == k
        mass = weights[keep].sum()
        score = float(np.average(scores[keep], weights=weights[keep])) if mass else None
        frequency = float(np.average(y[keep], weights=weights[keep])) if mass else None
        if mass:
            ece += mass / total * abs(score - frequency)
        table.append(
            {
                "lower": k / bins,
                "upper": (k + 1) / bins,
                "frames": int(keep.sum()),
                "weight": float(mass),
                "mean_defect_probability": score,
                "defect_frequency": frequency,
            }
        )
    return {
        "balanced_accuracy": (
            (good + defect) / 2 if good is not None and defect is not None else None
        ),
        "good_recall": good,
        "defect_recall": defect,
        "auroc": weighted_auc(y, scores, weights),
        "brier": float(np.average((scores - y) ** 2, weights=weights)),
        "ece": float(ece),
        "reliability_bins": table,
    }


def summarize_predictions(rows, score_key="score_raw", threshold=0.5, bins=10):
    if bins < 1:
        raise ValueError("ECE requires positive number of bins")
    rows, y, scores, weights = _arrays(rows, score_key, threshold)
    pred = scores >= threshold
    confusion = {
        "tn": int(((y == 0) & ~pred).sum()),
        "fp": int(((y == 0) & pred).sum()),
        "fn": int(((y == 1) & ~pred).sum()),
        "tp": int(((y == 1) & pred).sum()),
    }
    result = {
        "frames": len(rows),
        "parents": len({r["group"] for r in rows}),
        "threshold": threshold,
        "score_key": score_key,
        "confusion": confusion,
        "parent_weight_rule": "1 / all assessed frames in parent; class recall then average",
        "calibration_definition": "ECE is weighted probability-versus-defect-frequency, fixed equal-width bins",
        "recording_weighted": _values(y, scores, weights, threshold, bins),
        "frame_weighted": _values(y, scores, np.ones(len(rows)), threshold, bins),
    }
    result["recording_weighted_balanced_accuracy"] = result["recording_weighted"][
        "balanced_accuracy"
    ]
    result["recording_weighted_auroc"] = result["recording_weighted"]["auroc"]

    def subset(mask):
        yy, ss, ww = y[mask], scores[mask], weights[mask]
        pp = ss >= threshold
        return {
            "frames": int(mask.sum()),
            "parents": len({r["group"] for r, keep in zip(rows, mask) if keep}),
            "good_frames": int((yy == 0).sum()),
            "defect_frames": int((yy == 1).sum()),
            "confusion": {
                "tn": int(((yy == 0) & ~pp).sum()),
                "fp": int(((yy == 0) & pp).sum()),
                "fn": int(((yy == 1) & ~pp).sum()),
                "tp": int(((yy == 1) & pp).sum()),
            },
            "metrics": _values(yy, ss, ww, threshold, bins),
            "score_quantiles": (
                {str(q): float(np.quantile(ss, q)) for q in (0, 0.1, 0.5, 0.9, 1)}
                if len(ss)
                else {}
            ),
            "unsupported_reason": (
                None
                if len(set(yy)) == 2
                else "Both released classes required for BA/AUROC"
            ),
        }

    result["per_parent"] = {
        g: subset(np.array([r["group"] == g for r in rows]))
        for g in sorted({r["group"] for r in rows})
    }
    result["per_subtype"] = {
        str(label): {
            "released_label": label,
            **subset(np.array([r["label6"] == label for r in rows])),
        }
        for label in range(6)
    }
    result["per_condition_class"] = {
        f"{p}:{t}": subset(
            np.array([r["group"].split("-")[0] == p and r["target"] == t for r in rows])
        )
        for p, t in sorted({(r["group"].split("-")[0], r["target"]) for r in rows})
    }
    return result


import torch


def class_weights(rows):
    counts = Counter(r["group"] for r in rows)
    mass = [
        sum(1 / counts[r["group"]] for r in rows if r["target"] == label)
        for label in (0, 1)
    ]
    if not all(mass):
        raise ValueError("The quality probe requires both fitting classes")
    return torch.tensor(
        [len(rows) / (2 * mass[r["target"]] * counts[r["group"]]) for r in rows],
        dtype=torch.float32,
    )


def quality_metrics(rows, logits):
    logits = (
        logits.detach().double().cpu().numpy()
        if isinstance(logits, torch.Tensor)
        else np.asarray(logits, dtype=np.float64)
    )
    if logits.shape != (len(rows),) or not np.isfinite(logits).all():
        raise ValueError("One finite quality logit required per canonical image")
    scores = np.exp(-np.logaddexp(0.0, -logits))
    predictions = [
        {**row, "logit": float(logit), "score_raw": float(score)}
        for row, logit, score in zip(rows, logits, scores)
    ]
    report = summarize_predictions(predictions, threshold=0.5)
    y = np.asarray([r["target"] for r in rows])
    counts = Counter(r["group"] for r in rows)
    weights = np.asarray([1 / counts[r["group"]] for r in rows])
    report["recording_weighted"]["auroc"] = weighted_auc(y, logits, weights)
    report["frame_weighted"]["auroc"] = weighted_auc(y, logits, np.ones(len(rows)))
    report["recording_weighted_auroc"] = report["recording_weighted"]["auroc"]
    report["ranking_score"] = "raw logit, avoiding sigmoid saturation"
    for key, mass in (
        ("recording_weighted", weights),
        ("frame_weighted", np.ones(len(rows))),
    ):
        positive = logits >= 0
        tn, fp, fn, tp = [
            float(mass[mask].sum())
            for mask in (
                (y == 0) & ~positive,
                (y == 0) & positive,
                (y == 1) & ~positive,
                (y == 1) & positive,
            )
        ]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else None
        order = np.argsort(-logits, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(logits[order])), len(rows) - 1]
        true_positive = np.cumsum(mass[order] * y[order])[ends]
        all_positive = np.cumsum(mass[order])[ends]
        ap = (
            float(
                np.sum(
                    np.diff(np.r_[0.0, true_positive])
                    / mass[y == 1].sum()
                    * true_positive
                    / all_positive
                )
            )
            if mass[y == 1].sum()
            else None
        )
        report[key].update(
            precision=precision,
            f1=(
                2 * precision * recall / (precision + recall)
                if recall is not None and precision + recall
                else 0.0
            ),
            pr_auc=ap,
            false_alarm_rate=fp / (tn + fp) if tn + fp else None,
        )
    for label in range(1, 6):
        keep = np.asarray([r["label6"] == label for r in rows])
        report["per_subtype"][str(label)]["binary_defect_recall"] = (
            float(np.average(logits[keep] >= 0, weights=weights[keep]))
            if keep.any()
            else None
        )
    for group, values in report["per_parent"].items():
        keep = np.asarray([r["group"] == group for r in rows])
        values["metrics"]["auroc"] = weighted_auc(y[keep], logits[keep], weights[keep])
    return predictions, report


def selection_score(rows, logits):
    a = logits.detach().double().cpu().numpy()
    y = np.array([r["target"] for r in rows])
    counts = Counter(r["group"] for r in rows)
    w = np.array([1 / counts[r["group"]] for r in rows])
    p = a >= 0
    tpr = float(w[(y == 1) & p].sum() / w[y == 1].sum())
    tnr = float(w[(y == 0) & ~p].sum() / w[y == 0].sum())
    return {
        "selection_BA": (tpr + tnr) / 2,
        "selection_defect_recall": tpr,
        "selection_false_alarm_rate": 1 - tnr,
    }
