"""Distributed best-pair inference using the training implementation."""
from pathlib import Path
from weights import read_manifest, checkpoint_file
from ijepa.classifier import WeldClassifier
from ijepa.evaluate import predict_image
from ijepa.utils import configure_precision


def load_best(folder, device="cpu", assets=None):
    folder = Path(folder)
    inventory = read_manifest(folder)
    configure_precision()
    encoder = checkpoint_file(folder / "weights", inventory["artifacts"]["best_encoder"])
    classifier = checkpoint_file(folder / "weights", inventory["artifacts"]["best_classifier"])
    return WeldClassifier.from_files(encoder, classifier, device)


def predict(folder, image, device="cpu", assets=None):
    return predict_image(load_best(folder, device, assets), image, device)
