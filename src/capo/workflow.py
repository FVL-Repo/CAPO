import argparse
from dataclasses import replace
import json
import os

from capo.config import Config, local_path, load_mapping
from capo.online import run


def load_online(path):
    options = load_mapping(path)
    allowed = {
        "training_config",
        "robot_config",
        "norm_stats",
        "adapter",
        "actor_init",
        "reward_config",
        "episodes_per_round",
        "min_new_records",
        "train_processes",
        "training",
    }
    if set(options) - allowed:
        raise ValueError(f"Unknown online settings: {set(options) - allowed}")
    cfg = replace(Config.load(options["training_config"]), **options.get("training", {})).validate()
    if options.get("robot_config"):
        from capo.identity import bind_robot
        from capo.robots.flexiv_config import load_config

        os.environ["CAPO_FLEXIV_CONFIG"] = str(local_path(options["robot_config"]))
        bind_robot(cfg, load_config(options["robot_config"]))
    return cfg, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg, options = load_online(args.config)
    kwargs = {k: v for k, v in options.items() if k not in ("training_config", "training", "robot_config")}
    print(json.dumps(run(cfg, **kwargs), indent=2))


if __name__ == "__main__":
    main()
