import argparse
from pathlib import Path

import numpy as np

from capo.collect import make_record
from capo.config import Config, local_path
from capo.processing import Processor
from capo.replay import SOURCES, save_shard


def import_episode(path, cfg, processor, prompt):
    data = np.load(path, allow_pickle=False)
    actions = data["actions"]
    length = len(actions)
    for key in ("rewards", "terminated", "truncated", "source"):
        if data[key].shape != (length,):
            raise ValueError(f"{key} must have one entry per primitive action")
    if len(data["state"]) != length + 1:
        raise ValueError("state must include the final next observation (T+1 rows)")
    if length == 0 or not (data["terminated"][-1] or data["truncated"][-1]):
        raise ValueError("Episode must end in an explicit termination or truncation")
    if np.any(data["terminated"][:-1] | data["truncated"][:-1]):
        raise ValueError("Split multiple episodes into separate files")

    def obs(index):
        return {
            "state": data["state"][index],
            "prompt": prompt,
            "images": {k: data[f"image/{k}"][index] for k in cfg.image_keys if f"image/{k}" in data},
        }

    records, start = [], 0
    while start < length:
        source = str(data["source"][start])
        if source not in SOURCES:
            raise ValueError(f"Unknown source: {source}")
        end = start + 1
        while end < min(length, start + cfg.execution_horizon) and data["source"][end] == source:
            end += 1
        duration = end - start
        human = source != "autonomous"
        aligned = bool(data["pair_aligned"][start]) if "pair_aligned" in data else False
        if aligned and "proposals" not in data:
            raise ValueError("Aligned intervention requires a proposal recorded at that exact state")
        proposal = data["proposals"][start] if "proposals" in data else np.zeros_like(actions[start:end])
        result = {
            "actions": data["applied_actions"][start:end]
            if "applied_actions" in data
            else actions[start:end],
            "commands": actions[start:end],
            "rewards": data["rewards"][start:end],
            "next_obs": obs(end),
            "terminated": bool(data["terminated"][end - 1]),
            "truncated": bool(data["truncated"][end - 1]),
            "human_steps": duration if human else 0,
        }
        records.append(
            make_record(
                cfg,
                processor,
                obs(start),
                proposal,
                result,
                human=human,
                aligned=aligned,
                demonstration=source == "demonstration",
            )
        )
        start = end
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Project-local directory of NPZ episodes")
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mean-std", action="store_true", help="Use mean/std instead of pi_05 quantiles")
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    if (cfg.data_spec or {}).get("task", args.prompt) != args.prompt:
        raise ValueError("Import prompt differs from configured task")
    cfg.data_spec = {**(cfg.data_spec or {}), "task": args.prompt}
    processor = Processor(cfg, args.norm_stats, quantile=not args.mean_std)
    local_path(cfg.replay).mkdir(parents=True, exist_ok=True)
    cfg.save(local_path(cfg.replay) / "data_config.json")
    files = sorted(local_path(args.input).glob("*.npz"))
    if not files:
        raise ValueError("No NPZ episodes found")
    for file in files:
        records = import_episode(file, cfg, processor, args.prompt)
        output = local_path(cfg.replay) / (Path(file).stem + ".pt")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing replay shard {output}")
        save_shard(output, records, cfg)
        print(f"{file.name}: {len(records)} executed transitions -> {output}")


if __name__ == "__main__":
    main()
