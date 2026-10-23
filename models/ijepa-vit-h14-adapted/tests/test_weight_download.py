import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def fixture(tmp_path):
    source = Path(__file__).resolve().parents[1] / "weights/download.py"
    spec = importlib.util.spec_from_file_location("this_model_weights", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pieces, whole = [], b""
    for index, data in enumerate((b"first part", b"second part")):
        file = tmp_path / str(index)
        file.write_bytes(data)
        pieces.append(
            {
                "asset": file.name,
                "url": file.as_uri(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        whole += data
    manifest = {
        "artifacts": {
            "encoder": {
                "required": True,
                "path": "encoder.pt",
                "bytes": len(whole),
                "sha256": hashlib.sha256(whole).hexdigest(),
                "parts": pieces,
            }
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return module, path, manifest, whole, tmp_path / "downloaded"


def test_multipart_checkpoint_is_reassembled_and_reverified(fixture):
    module, path, _, whole, output = fixture
    assert module.download(path, output) == {"encoder.pt": "downloaded_and_verified"}
    assert (output / "encoder.pt").read_bytes() == whole
    assert module.download(path, output, verify_only=True) == {"encoder.pt": "verified"}


def test_changed_part_never_becomes_a_final_checkpoint(fixture):
    module, path, manifest, _, output = fixture
    manifest["artifacts"]["encoder"]["parts"][1]["sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="part checksum"):
        module.download(path, output)
    assert not (output / "encoder.pt").exists()
    assert not (output / "encoder.pt.partial").exists()


def test_whole_file_checksum_is_checked_independently(fixture):
    module, path, manifest, _, output = fixture
    manifest["artifacts"]["encoder"]["sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Reassembled"):
        module.download(path, output)


def test_missing_and_changed_local_weights_fail_closed(fixture):
    module, path, _, _, output = fixture
    with pytest.raises(FileNotFoundError):
        module.download(path, output, verify_only=True)
    output.mkdir(exist_ok=True)
    (output / "encoder.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing checkpoint"):
        module.download(path, output)


def test_nested_checkpoint_and_identical_encoder_are_retained(fixture):
    module, path, manifest, whole, output = fixture
    item=manifest["artifacts"]["encoder"]
    item["path"]="best/encoder.pt"
    manifest["artifacts"]["final_encoder"]={**item,"path":"final/encoder.pt"}
    path.write_text(json.dumps(manifest))
    result=module.download(path,output)
    assert result["final/encoder.pt"]=="verified_local_hardlink"
    assert (output/"best/encoder.pt").read_bytes()==whole
    assert (output/"final/encoder.pt").read_bytes()==whole
    assert all(v=="verified" for v in module.download(path,output,verify_only=True).values())


def test_checkpoint_path_cannot_escape_weights_folder(fixture):
    module,path,manifest,_,output=fixture
    manifest["artifacts"]["encoder"]["path"]="../outside.pt"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="inside the weights"):
        module.download(path,output)
    assert not (output.parent/"outside.pt").exists()
