"""Download actual release assets into this model's weights folder and verify bytes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(manifest_path, output=None, verify_only=False, only=None):
    manifest_path = Path(manifest_path)
    folder = Path(output) if output else manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    available = [v for v in manifest["artifacts"].values() if v.get("required")]
    if only and only not in {v["path"] for v in available}:
        raise ValueError("No required checkpoint with this filename")
    result = {}
    verified = {}
    for item in available:
        if only and item["path"] != only:
            continue
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Checkpoint filename must stay inside the weights folder")
        target = folder / relative
        if not target.resolve().is_relative_to(folder.resolve()) or target.is_symlink():
            raise ValueError("Checkpoint filename must stay inside the weights folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if (
                target.stat().st_size != item["bytes"]
                or sha256(target) != item["sha256"]
            ):
                raise ValueError("Existing checkpoint differs: " + str(target))
            result[item["path"]] = "verified"
            verified[item["sha256"]] = target
            continue
        if verify_only:
            raise FileNotFoundError("Checkpoint is missing: " + str(target))
        if item["sha256"] in verified:
            # Best/final frozen encoders have identical bytes. Keep complete
            # usable files in both folders without downloading another copy.
            os.link(verified[item["sha256"]], target)
            result[item["path"]] = "verified_local_hardlink"
            continue
        temporary = target.with_name(target.name + ".partial")
        whole = hashlib.sha256()
        try:
            with temporary.open("wb") as out:
                for part in item["parts"]:
                    start = out.tell()
                    previous = whole.copy()
                    for attempt in range(3):
                        try:
                            out.seek(start)
                            out.truncate()
                            whole = previous.copy()
                            actual, size = hashlib.sha256(), 0
                            request = urllib.request.Request(
                                part["url"],
                                headers={"User-Agent": "TIG-Assessment-weights"},
                            )
                            with urllib.request.urlopen(request, timeout=60) as stream:
                                while True:
                                    chunk = stream.read(8 << 20)
                                    if not chunk:
                                        break
                                    out.write(chunk)
                                    actual.update(chunk)
                                    whole.update(chunk)
                                    size += len(chunk)
                            if (
                                size != part["bytes"]
                                or actual.hexdigest() != part["sha256"]
                            ):
                                raise ValueError(
                                    "Release part checksum/size mismatch: "
                                    + part["asset"]
                                )
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                if out.tell() != item["bytes"] or whole.hexdigest() != item["sha256"]:
                    raise ValueError("Reassembled checkpoint checksum/size mismatch")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        result[item["path"]] = "downloaded_and_verified"
        verified[item["sha256"]] = target
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--only")
    args = parser.parse_args()
    result = download(
        Path(__file__).with_name("manifest.json"),
        args.output_dir,
        args.verify_only,
        args.only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
