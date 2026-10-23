"""Submission packages expose the original validation-selected checkpoint."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import torch


def test_package_contains_only_the_reported_best_checkpoint():
    folder = Path(__file__).resolve().parents[1]
    weights = folder / 'weights'
    inventory = json.loads((weights / 'manifest.json').read_text())
    assert set(inventory['checkpoints']) == {'best'}
    assert set(inventory['artifacts']) == {'best_encoder', 'best_classifier'}
    assert not (weights/'final').exists()
    best = inventory['checkpoints']['best']
    desc = json.loads((weights / best['descriptor']).read_text())
    best_head = torch.load(weights / inventory['artifacts']['best_classifier']['path'], map_location='cpu', weights_only=True)
    assert best_head['epoch'] == desc['classifier_epoch']
    assert desc['provenance']['status'] == 'original_saved'


def test_unavailable_final_checkpoint_is_rejected_without_fallback(tmp_path):
    folder = Path(__file__).resolve().parents[1]
    shutil.copyfile(folder / 'run.py', tmp_path / 'run.py')
    (tmp_path / 'model.json').write_text(json.dumps({'entrypoint': 'ijepa'}))
    (tmp_path / 'weights').mkdir()
    (tmp_path / 'weights/manifest.json').write_text(json.dumps({'checkpoints': {'best': {}}}))
    result = subprocess.run([sys.executable, str(tmp_path / 'run.py'), 'predict-local', '--image', 'unused.png', '--checkpoint', 'final'], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr
