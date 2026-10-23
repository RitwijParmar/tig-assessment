"""Seal completed configurations after an explicit pre-assessment search stop.

This command performs no optimizer updates. It requires completed candidate
audits, preserves the original recipe and records the changed search scope.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tig_benchmark.assets import load_assets, model_config
from tig_benchmark.checkpoints import export_selected
from tig_benchmark.pipeline import (
    implementation_identity, load_data_payload, tensor_identity, verify,
)
from tig_benchmark.models import build
from tig_benchmark.shared import atomic_json, configure_precision, digest, sha256


def finalize(run, rates, reason, device="cuda"):
    run = Path(run).resolve()
    for name in ("seal.json", ".assessment.started", "assessment_metrics.json", "selected", "final"):
        if (run / name).exists():
            raise ValueError("Finalization requires an unsealed, unassessed run without exports")
    cfg = json.loads((run / "config.json").read_text())
    _, roles = load_data_payload(cfg)
    spec = model_config(cfg["model"])
    if cfg["model"] != "mobilenetv5" or not rates or len(rates) != len(set(rates)):
        raise ValueError("Supply distinct completed MobileNet encoder rates")
    if any(rate not in spec["supervised"]["encoder_lr_candidates"] for rate in rates):
        raise ValueError("Rate was not in the original recipe")
    lock = load_assets(cfg["model"], run / "assets.local.json")
    frozen = torch.load(run / "frozen/best.pt", map_location="cpu", weights_only=True)
    chosen = dict(mode="frozen", epoch=frozen["epoch"], score=frozen["score"],
                  head_path=str(run / "frozen/best.pt"), encoder_path=None)
    completed = []
    for rate in rates:
        folder = run / "supervised" / f"lr{rate:g}"
        audit = json.loads((folder / "weight_audit.json").read_text())
        history = json.loads((folder / "history.json").read_text())
        best = max(history, key=lambda item: item["selection_BA"])
        stale = len(history) - 1 - best["epoch"]
        if history[-1]["epoch"] < spec["supervised"]["epochs"] and stale < spec["supervised"]["patience"]:
            raise ValueError("Candidate did not finish its declared horizon/stopping rule")
        head = torch.load(folder / "best_head.pt", map_location="cpu", weights_only=True)
        final = torch.load(folder / "final_head.pt", map_location="cpu", weights_only=True)
        assert final["epoch"] == history[-1]["epoch"]
        assert head["score"]["selection_BA"] == best["selection_BA"]
        assert head["epoch"] == (best["epoch"] or frozen["epoch"])
        assert head["fitting_rows_sha256"] == digest(roles["fitting"])
        assert head["selection_rows_sha256"] == digest(roles["selection"])
        assert audit["epoch0_eligible"] and audit["optimizer_updates"] > 0
        for kind, probe in (("best", head), ("final", final)):
            if kind == "best" and best["epoch"] == 0:
                continue
            state = torch.load(folder / (kind + "_encoder.pt"), map_location="cpu", weights_only=True)
            assert state["identity"] == probe["identity"]
            del state
        completed.append(dict(rate=rate, completed_epoch=final["epoch"], best_epoch=best["epoch"],
                              best_validation_BA=best["selection_BA"]))
        if best["selection_BA"] > chosen["score"]["selection_BA"]:
            chosen = dict(mode="supervised", epoch=best["epoch"], score=head["score"], rate=rate,
                          head_path=str(folder / "best_head.pt"), encoder_path=str(folder / "best_encoder.pt"))
    amendment = dict(
        utc=datetime.now(timezone.utc).isoformat(), reason=reason,
        assessment_opened=False, originally_declared_rates=spec["supervised"]["encoder_lr_candidates"],
        completed_rates_used_for_selection=rates, completed_candidates=completed,
        skipped_rates=[rate for rate in spec["supervised"]["encoder_lr_candidates"] if rate not in rates],
        selection_rule="Compare the frozen baseline and completed configurations only, using validation; earliest ties retained.",
    )
    atomic_json(run / "selection_amendment.json", amendment)
    configure_precision()
    encoder, _ = build(lock, torch.device(device), cfg["geometry"])
    before = tensor_identity(encoder)
    del encoder
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    chosen = export_selected(run, chosen, lock, cfg)
    seal = dict(
        format="tig-transfer-selection-v1", selected=chosen, threshold=0.5,
        config_sha256=sha256(run / "config.json"), assets_sha256=sha256(run / "assets.local.json"),
        implementation_identity=implementation_identity(), role_sha256={key: digest(value) for key, value in roles.items()},
        head_sha256=sha256(chosen["head_path"]), replay_sha256=sha256(run / "replay.local.pt"),
        encoder_sha256=sha256(chosen["encoder_path"]) if chosen["encoder_path"] else None,
        frozen_encoder_tensor_sha256=before, assessment_accessed_for_selection=False,
        assessment_previously_inspected=True, selection_amendment=amendment,
        selection_amendment_sha256=sha256(run / "selection_amendment.json"),
    )
    atomic_json(run / "seal.json", seal)
    replay = verify(run, device)
    atomic_json(run / "status.json", dict(stage="sealed_and_reload_verified", assessment_completed=False))
    return dict(selected=chosen, amendment=amendment, reload=replay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(finalize(args.run, args.rates, args.reason, args.device), indent=2))
