"""An unavailable historical final checkpoint must not silently load best."""
from pathlib import Path
import subprocess
import sys


def test_unretained_final_checkpoint_is_rejected():
    folder=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(folder/'run.py'),'predict-local','--image','unused.png','--checkpoint','final'],text=True,capture_output=True)
    assert result.returncode==2
    assert 'not retained' in result.stderr
