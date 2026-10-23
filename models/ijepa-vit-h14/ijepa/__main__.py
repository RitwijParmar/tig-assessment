"""Run `python -m ijepa --help`; no model construction happens on import."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="I-JEPA ViT-H/14: native I-JEPA adaptation and linear readout"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Validate local metadata and write local paths"
    )
    prepare.add_argument(
        "--metadata",
        required=True,
        help="Local canonical CSV or original registration JSON",
    )
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--base-checkpoint", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser(
        "train", help="Adapt with JEPA and fit mean-pooled linear readouts"
    )
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda")
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate the sealed selected model on assessment"
    )
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--device", default="cuda")
    predict = commands.add_parser(
        "predict", help="Predict good/defect for a single image"
    )
    predict.add_argument("--run", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        print(prepare(args.metadata, args.data_root, args.base_checkpoint, args.output))
    elif args.command == "train":
        from .train import train

        print(json.dumps(train(args.config, args.output, args.device), indent=2))
    elif args.command == "evaluate":
        from .evaluate import evaluate

        print(json.dumps(evaluate(args.run, args.device), indent=2))
    else:
        from .utils import configure_precision
        from .classifier import WeldClassifier
        from .evaluate import predict_image

        configure_precision()
        model = WeldClassifier.from_run(args.run, args.device)
        print(json.dumps(predict_image(model, args.image, args.device), indent=2))


if __name__ == "__main__":
    main()
