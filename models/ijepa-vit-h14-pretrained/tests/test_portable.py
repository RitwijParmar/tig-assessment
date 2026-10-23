"""Final pairs retain distinct last heads and never silently fall back to best."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import torch


def test_final_pair_has_explicit_reconstruction_and_distinct_last_head():
    folder = Path(__file__).resolve().parents[1]
    weights = folder / 'weights'
    inventory = json.loads((weights / 'manifest.json').read_text())
    final = inventory['checkpoints']['final']
    desc = json.loads((weights / final['descriptor']).read_text())
    best_head = torch.load(weights / inventory['artifacts']['best_classifier']['path'], map_location='cpu', weights_only=True)
    final_head = torch.load(weights / inventory['artifacts'][final['classifier']]['path'], map_location='cpu', weights_only=True)
    assert final_head['epoch'] == desc['classifier_epoch'] > best_head['epoch']
    assert not torch.equal(final_head['head']['weight'], best_head['head']['weight'])
    assert torch.equal(final_head['mean'], best_head['mean'])
    assert torch.equal(final_head['std'], best_head['std'])
    assert desc['provenance']['status'] == 'reconstructed_cpu_replay'
    assert desc['assessment_evaluated'] is False
    assert final_head['reconstruction']['bitwise_original_final_identity_claimed'] is False


def test_unavailable_final_checkpoint_is_rejected_without_fallback(tmp_path):
    folder = Path(__file__).resolve().parents[1]
    shutil.copyfile(folder / 'run.py', tmp_path / 'run.py')
    (tmp_path / 'model.json').write_text(json.dumps({'entrypoint': 'ijepa'}))
    (tmp_path / 'weights').mkdir()
    (tmp_path / 'weights/manifest.json').write_text(json.dumps({'checkpoints': {'best': {}}}))
    result = subprocess.run([sys.executable, str(tmp_path / 'run.py'), 'predict-local', '--image', 'unused.png', '--checkpoint', 'final'], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'not retained' in result.stderr
