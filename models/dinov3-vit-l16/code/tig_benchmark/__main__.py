"""Run from the repository root: python -m tig_benchmark --help."""

import argparse
import json
from . import data, assets, pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Exact final-I-JEPA split, released MobileNetV5/DINOv3 encoders"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--metadata", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p = subs.add_parser("register-assets")
    p.add_argument("--model", choices=assets.MODEL_DIRS, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = subs.add_parser("fetch-assets")
    p.add_argument("--model", choices=assets.MODEL_DIRS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-only", action="store_true")
    p = subs.add_parser("train")
    p.add_argument("--model", choices=assets.MODEL_DIRS, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--geometry", choices=("native", "letterbox224"), default="native")
    for name in ("verify", "evaluate", "predict"):
        p = subs.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--device", default="cpu")
        if name == "predict":
            p.add_argument("--image", required=True)
    a = parser.parse_args()
    if a.command == "prepare":
        result = data.prepare(a.metadata, a.data_root, a.output)
    elif a.command == "register-assets":
        result = assets.register(a.model, a.source, a.checkpoint, a.output)
    elif a.command == "fetch-assets":
        from .download import fetch

        result = fetch(a.model, a.output, source_only=a.source_only)
    elif a.command == "train":
        result = pipeline.fit(
            a.model,
            a.data,
            a.assets,
            a.output,
            a.device,
            a.batch_size,
            a.workers,
            a.geometry,
            False,
        )
    elif a.command == "verify":
        result = pipeline.verify(a.run, a.device)
    elif a.command == "evaluate":
        result = pipeline.evaluate(a.run, a.device)
    else:
        result = pipeline.predict(a.run, a.image, a.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
