"""Atomic checkpoints and reproducible identities."""

from contextlib import contextmanager
import hashlib, json, os, random, tempfile
from pathlib import Path
import numpy as np
import torch


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_bytes(path, raw)


def _atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    os.close(fd)
    try:
        torch.save(payload, name)
        reopened = torch.load(name, map_location="cpu", weights_only=True)
        if reopened.get("identity") != payload.get("identity"):
            raise ValueError("Checkpoint identity did not survive saved-file reload")
        with open(name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
        return sha256(path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def exclusive_run(directory):
    """An OS advisory lock is released even after a process interruption."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "This run is already owned by another live process"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def rng_state():
    ns = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state(
        (ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError(
                "CUDA RNG device count differs; declare a migration rather than silently resuming"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def configure_precision():
    """Match the recorded float32/highest-matmul/default-TF32-convolution recipe."""
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends.cudnn, "conv"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        torch.backends.cudnn.allow_tf32 = True
