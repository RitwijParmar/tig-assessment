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
    p.add_argument("--source-only", action="store_true", help="Set up architecture source for best-checkpoint inference")
    p = subs.add_parser("train")
    p.add_argument("--model", choices=assets.MODEL_DIRS, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--geometry", choices=("native", "letterbox224"), default="native")
    p.add_argument("--finetune", action="store_true", default=True, help="Train the recorded full-encoder recipe (default)")
    p.add_argument("--encoder-lr", type=float, default=1e-5, choices=(1e-5, 3e-5, 1e-4), help="One declared encoder LR; the submitted model uses 1e-5")
    p.add_argument("--resume", action="store_true", help="Continue an unsealed run from complete saved epoch checkpoints")
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
            a.finetune,
            a.resume,
            encoder_lr=a.encoder_lr,
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
