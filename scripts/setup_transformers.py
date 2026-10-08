import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import sys


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Install CAPO in a virtual environment before applying the OpenPI patches")
    package = importlib.metadata.distribution("transformers")
    if package.version != "4.53.2":
        raise RuntimeError("OpenPI requires transformers==4.53.2")
    source = (
        Path(__file__).resolve().parents[1]
        / "third_party/openpi/src/openpi/models_pytorch/transformers_replace"
    )
    destination = Path(package.locate_file("transformers"))
    for path in sorted(source.rglob("*.py")):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise RuntimeError(f"Failed to install patch: {target.name}")
    print("Installed OpenPI patches for Transformers 4.53.2")


if __name__ == "__main__":
    main()
