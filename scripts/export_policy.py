import argparse

import torch
from safetensors.torch import save_model

from capo.backbone import make_backbone
from capo.config import Config, local_path
from capo.checkpoints import load_actor_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch.load(local_path(args.checkpoint), map_location="cpu", weights_only=True)
    saved_config = dict(payload["config"])
    saved_config.setdefault("actor_tuning", "full")
    cfg = Config(**saved_config).validate()
    cfg.device = "cpu"
    if cfg.backend != "pi05":
        raise ValueError("Only real pi_05 checkpoints can be exported")
    actor = make_backbone(cfg)
    load_actor_state(actor, payload["actor"], payload["config"])
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    actor.merge_lora_for_export()
    save_model(actor.model, str(output / "model.safetensors"))
    cfg.checkpoint = str((output / "model.safetensors").relative_to(local_path(".")))
    from capo.identity import file_sha256

    cfg.base_sha256 = file_sha256(output / "model.safetensors")
    cfg.save(output / "capo_config.json")
    print(output / "model.safetensors")


if __name__ == "__main__":
    main()
