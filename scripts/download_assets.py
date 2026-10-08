"""Download the tokenizer and frozen ResNet encoder; actor weights are downloaded separately."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "tokenizer": (
        "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model",
        "paligemma_tokenizer.model",
        "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",
    ),
    "resnet": (
        "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl",
        "resnet10_params.pkl",
        "175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b",
    ),
}


def download(name):
    url, filename, checksum = ASSETS[name]
    destination = ROOT / "assets" / filename
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Download checksum mismatch: {filename}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Existing asset checksum mismatch: {filename}")
    if name == "resnet":
        subprocess.run([sys.executable, str(ROOT / "scripts/convert_resnet10.py")], cwd=ROOT, check=True)
    print(f"Ready: {destination.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(ASSETS))
    args = parser.parse_args()
    for name in [args.only] if args.only else ASSETS:
        download(name)


if __name__ == "__main__":
    main()
