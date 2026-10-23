"""Run this model independently from its own directory."""

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "weights"))
meta = json.loads((ROOT / "model.json").read_text())
command = sys.argv[1] if len(sys.argv) > 1 else "--help"
if command in ("fetch-weights", "verify-weights"):
    from download import main

    sys.argv = (
        [sys.argv[0]]
        + (["--verify-only"] if command == "verify-weights" else [])
        + sys.argv[2:]
    )
    main()
elif command == "test":
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests"), *sys.argv[2:]],
            cwd=ROOT,
        )
    )
elif command == "predict-local":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", choices=("best", "final"), default="best")
    p.add_argument("--assets")
    a = p.parse_args(sys.argv[2:])
    if meta["entrypoint"] == "tig_benchmark":
        from tig_benchmark.inference import predict_local

        print(
            json.dumps(
                predict_local(ROOT, a.image, a.device, a.checkpoint, a.assets), indent=2
            )
        )
        raise SystemExit(0)
    inventory = json.loads((ROOT / "weights/manifest.json").read_text())
    if a.checkpoint not in inventory["checkpoints"]:
        p.error("This checkpoint was not retained; use --checkpoint best")
    pair = inventory["checkpoints"][a.checkpoint]
    manifest = inventory["artifacts"]
    encoder_spec = manifest[pair["encoder"]]
    classifier_spec = manifest[pair["classifier"]]
    from ijepa.classifier import WeldClassifier
    from ijepa.evaluate import predict_image
    from ijepa.utils import configure_precision

    configure_precision()
    model = WeldClassifier.from_files(
        ROOT / "weights" / encoder_spec["path"],
        ROOT / "weights" / classifier_spec["path"],
        a.device,
        expected={
            "encoder": encoder_spec["sha256"],
            "classifier": classifier_spec["sha256"],
        },
    )
    print(json.dumps(predict_image(model, a.image, a.device), indent=2))
else:
    family = meta["entrypoint"]
    if (
        family == "tig_benchmark"
        and command in ("train", "register-assets", "fetch-assets")
        and "--model" not in sys.argv
    ):
        sys.argv += ["--model", meta["runtime_model"]]
    importlib.import_module(family + ".__main__").main()
