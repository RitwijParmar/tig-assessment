"""Training, evaluation and best-checkpoint inference for this model."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from weights import download, read_manifest, sha256


def main():
    inventory = read_manifest(ROOT)
    family = inventory["inference"]["family"]
    command = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if command in {"prepare", "train", "evaluate", "register-assets", "fetch-assets", "verify", "predict-run"}:
        package = "ijepa" if family == "ijepa" else "tig_benchmark"
        if command == "predict-run":
            sys.argv[1] = "predict"
        if family != "ijepa" and command in {"train", "register-assets", "fetch-assets"} and "--model" not in sys.argv:
            sys.argv += ["--model", family]
        importlib.import_module(package + ".__main__").main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch-weights", help="Download and verify the distributed best pair")
    commands.add_parser("verify-weights", help="Verify local best encoder and classifier bytes")
    p = commands.add_parser("fetch-source", help="Install pinned source for best-checkpoint inference")
    p.add_argument("--output")
    p = commands.add_parser("predict", help="Predict using the distributed best checkpoint")
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--assets")
    p = commands.add_parser("evaluate-best", help="Evaluate distributed best weights on the registered assessment split")
    p.add_argument("--data", required=True, help="Prepared config.local.json (I-JEPA) or data.local.json (native models)")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--assets")
    p.add_argument("--batch-size", type=int, default=2)
    commands.add_parser("prepare", help="Validate canonical data and the exact recording split")
    commands.add_parser("train", help="Train the registered model recipe; use train --help")
    commands.add_parser("evaluate", help="Assess a sealed training run; use evaluate --help")
    commands.add_parser("predict-run", help="Predict from a sealed training run")
    if family != "ijepa":
        commands.add_parser("fetch-assets", help="Fetch original pretrained assets for training")
        commands.add_parser("register-assets", help="Register already-downloaded training assets")
        commands.add_parser("verify", help="Reload and verify a sealed training run")
    commands.add_parser("test", help="Run training, isolation and checkpoint regression checks")
    args = parser.parse_args()
    if args.command in {"fetch-weights", "verify-weights"}:
        result = download(ROOT / "weights/manifest.json", verify_only=args.command == "verify-weights")
    elif args.command == "evaluate-best":
        from evaluation import evaluate_best
        result = evaluate_best(ROOT, args.data, args.output, args.device, args.assets, args.batch_size)
    elif args.command == "test":
        raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT))
    elif args.command == "fetch-source":
        if family == "ijepa":
            folder = ROOT / "code/ijepa/vendor"
            pin = json.loads((folder / "SOURCE_MANIFEST.json").read_text())
            if not all(sha256(folder / item["file"]) == item["local_sha256"] for item in pin["files"]):
                raise ValueError("Bundled I-JEPA source changed")
            result = {"source": "bundled", "verified": True}
        else:
            from tig_benchmark.download import fetch
            result = fetch(family, args.output or ROOT / "assets/source", source_only=True)
    else:
        from inference import predict
        result = predict(ROOT, args.image, args.device, args.assets)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
