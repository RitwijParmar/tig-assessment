"""Run each model's tests in its own process to check package independence."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for folder in sorted((root / "models").iterdir()):
    print("Checking " + folder.name, flush=True)
    subprocess.run([sys.executable, "run.py", "test"], cwd=folder, check=True)
