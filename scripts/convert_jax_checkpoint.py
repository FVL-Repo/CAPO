import argparse
import importlib.util
import sys
import json

from capo.backbone import use_vendored_openpi
from capo.config import Config, ROOT, local_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/workflow/sft.yaml")
    parser.add_argument("--input", required=True, help="Copied OpenPI checkpoint containing params/")
    parser.add_argument("--output", default="assets/openpi_pi05_base")
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    if (local_path(args.output) / "model.safetensors").exists():
        raise FileExistsError("Converted weights already exist; choose a new output directory")
    use_vendored_openpi()
    path = ROOT / "third_party/openpi/examples/convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("capo_upstream_converter", path)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    from openpi.models.pi0_config import Pi0Config

    model_config = Pi0Config(
        pi05=True,
        action_horizon=cfg.action_horizon,
        action_dim=cfg.action_dim,
        max_token_len=cfg.max_token_len,
        dtype=cfg.precision,
        pytorch_compile_mode=None,
    )
    converter.convert_pi0_checkpoint(
        str(local_path(args.input)), cfg.precision, str(local_path(args.output)), model_config
    )
    from capo.identity import file_sha256

    provenance = {
        "input": str(local_path(args.input)),
        "converter": str(path.relative_to(ROOT)),
        "precision": cfg.precision,
        "sha256": file_sha256(local_path(args.output) / "model.safetensors"),
        "input_files": [
            {"path": str(p.relative_to(local_path(args.input))), "sha256": file_sha256(p)}
            for p in sorted(local_path(args.input).rglob("*"))
            if p.is_file()
        ],
    }
    (local_path(args.output) / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
