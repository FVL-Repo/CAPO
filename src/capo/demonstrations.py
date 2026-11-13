import argparse
from collections import OrderedDict
from contextlib import nullcontext
import json
import os
import uuid

import numpy as np
import torch

from capo.batching import stack_trees
from capo.config import Config, local_path, load_mapping
from capo.identity import check_spec, file_sha256
from capo.processing import Processor


def dataset_config(path):
    options = load_mapping(path)
    allowed = {
        "training_config",
        "robot_config",
        "adapter",
        "input",
        "output",
        "norm_stats",
        "task",
        "quantile",
    }
    if set(options) - allowed:
        raise ValueError(f"Unknown dataset fields: {set(options) - allowed}")
    cfg = Config.load(options["training_config"])
    if options.get("robot_config"):
        from capo.identity import bind_robot
        from capo.robots.flexiv_config import load_config

        os.environ["CAPO_FLEXIV_CONFIG"] = str(local_path(options["robot_config"]))
        bind_robot(cfg, load_config(options["robot_config"]))
    spec = dict(cfg.data_spec or {})
    if spec.get("task", options["task"]) != options["task"]:
        raise ValueError("Dataset task differs from robot/training task")
    cfg.data_spec = {**spec, "task": options["task"]}
    cfg.demonstrations = options["output"]
    return cfg, options


def raw_samples(path, cfg):
    with np.load(path, allow_pickle=False) as data:
        states, actions = data["state"], data["actions"]
        if states.ndim != 2 or not 0 < states.shape[1] <= cfg.state_dim:
            raise ValueError(f"Invalid states in {path}")
        if actions.ndim not in (2, 3) or len(actions) != len(states) or not len(states):
            raise ValueError(f"Require equal nonempty state/action sample counts in {path}")
        if (
            actions.shape[-1] != cfg.robot_action_dim
            or not np.isfinite(states).all()
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Invalid action coordinates in {path}")
        metadata = json.loads(str(data["robot_metadata"])) if "robot_metadata" in data else {}
        prompt = str(data["prompt"]) if "prompt" in data else cfg.data_spec["task"]
        images = {k: data["image/" + k] for k in cfg.image_keys if "image/" + k in data}
        for camera, image in images.items():
            if len(image) != len(states) or image.dtype != np.uint8:
                raise ValueError(f"Camera/sample mismatch in {path}: {camera}")
        for i in range(len(states)):
            if actions.ndim == 2:
                commands = actions[i : i + cfg.action_horizon]
            else:
                length = int(data["lengths"][i]) if "lengths" in data else actions.shape[1]
                if not 1 <= length <= min(actions.shape[1], cfg.action_horizon):
                    raise ValueError("Invalid demonstration chunk length")
                commands = actions[i, :length]
            yield (
                {
                    "state": states[i],
                    "prompt": prompt,
                    "robot_metadata": metadata,
                    "images": {k: v[i] for k, v in images.items()},
                },
                commands,
            )


def collect_raw(cfg, options, robot, episodes):
    output = local_path(options["input"])
    output.mkdir(parents=True, exist_ok=True)
    if hasattr(robot, "demonstration"):
        robot.demonstration = True
    count = 0
    for _ in range(episodes):
        episode = uuid.uuid4().hex
        obs = robot.reset()
        pending = None

        def flush_raw():
            nonlocal pending
            if pending is None:
                return
            path, arrays = pending
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            path.with_suffix(".tmp").replace(path)
            pending = None

        try:
            while True:
                selected, human, _ = robot.choose_action(
                    obs, np.zeros((cfg.execution_horizon, cfg.robot_action_dim), np.float32)
                )
                if not human:
                    raise ValueError("SFT collection requires human control")
                result = robot.execute(selected)
                if result.get("quarantined") or result.get("safety_abort"):
                    if pending is not None:
                        pending[1]["truncated"][0] = True
                        flush_raw()
                    raise RuntimeError("Unconfirmed execution; inspect robot journal before continuing")
                if result.get("boundary") and pending is not None:
                    arrays = pending[1]
                    if result.get("reward_override") is not None:
                        arrays["rewards"][0, -1] = result["reward_override"]
                    arrays["terminated"][0] = result["terminated"]
                    arrays["truncated"][0] = result["truncated"]
                    arrays["next_state"] = np.asarray(result["next_obs"]["state"], np.float32)[None]
                    for k, v in result["next_obs"]["images"].items():
                        arrays["next_image/" + k] = np.asarray(v)[None]
                if not result.get("retry") and not result.get("boundary"):
                    flush_raw()
                    commands = np.asarray(result.get("commands", result["actions"]), np.float32)
                    if not 1 <= len(commands) <= cfg.action_horizon or result["human_steps"] != len(commands):
                        raise ValueError("Expected confirmed human command chunk")
                    arrays = {
                        "state": np.asarray(obs["state"], np.float32)[None],
                        "actions": commands[None],
                        "applied_actions": np.asarray(result["actions"], np.float32)[None],
                        "lengths": np.asarray([len(commands)]),
                        "prompt": np.asarray(obs["prompt"]),
                        "robot_metadata": np.asarray(json.dumps(obs.get("robot_metadata", {}))),
                        **{"image/" + k: np.asarray(v)[None] for k, v in obs["images"].items()},
                        "next_state": np.asarray(result["next_obs"]["state"], np.float32)[None],
                        "rewards": np.asarray(result["rewards"], np.float32)[None],
                        "terminated": np.asarray([result["terminated"]], dtype=bool),
                        "truncated": np.asarray([result["truncated"]], dtype=bool),
                        **{"next_image/" + k: np.asarray(v)[None] for k, v in result["next_obs"]["images"].items()},
                    }
                    path = output / f"{episode}_{count:08d}.npz"
                    pending = path, arrays
                    count += 1
                if result.get("terminated") or result.get("truncated"):
                    flush_raw()
                    break
                obs = result["next_obs"]
        except BaseException:
            if pending is not None:
                pending[1]["truncated"][0] = True
                flush_raw()
            raise
    return count


def raw_transition(path, index, raw, commands, cfg, processor, *, archive=None):
    """Optional real TD transition alongside a supervised sample; never infer missing rewards."""
    from capo.collect import make_record

    fields = {"next_state", "rewards", "terminated", "truncated"}
    with (nullcontext(archive) if archive is not None else np.load(path, allow_pickle=False)) as data:
        present = fields.intersection(data)
        if not present:
            return None
        if present != fields or data["actions"].ndim != 3:
            raise ValueError("TD demonstrations require chunked actions and complete successor/reward/boundary fields")
        cameras = {k: data["next_image/" + k][index] for k in raw["images"]}
        result = {
            "actions": data["applied_actions"][index, :len(commands)] if "applied_actions" in data else commands,
            "commands": commands,
            "rewards": data["rewards"][index, :len(commands)],
            "next_obs": {**raw, "state": data["next_state"][index], "images": cameras},
            "human_steps": len(commands), "terminated": bool(data["terminated"][index]),
            "truncated": bool(data["truncated"][index]),
            "success": False, "safety_abort": False,
        }
        
        record = make_record(cfg, processor, raw, commands, result,
                             human=True, aligned=False, demonstration=True)
        record["metadata"].pop("success", None)
        if "applied_actions" not in data:
            record.pop("applied_actions", None)
        return record


def prepare(cfg, options, *, existing_norm=False):
    files = sorted(local_path(options["input"]).glob("*.npz"))
    if not files:
        raise ValueError("No raw demonstration NPZ files")
    output = local_path(options["output"])
    if (output / "manifest.json").exists():
        raise FileExistsError("Prepared demonstrations are immutable; select a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    stats_path = local_path(options["norm_stats"])
    if not existing_norm:
        from capo.backbone import use_vendored_openpi

        use_vendored_openpi()
        from openpi.shared.normalize import RunningStats, serialize_json

        stats = {key: RunningStats() for key in ("state", "actions")}

        for path in files:
            for raw, commands in raw_samples(path, cfg):
                stats["state"].update(np.asarray(raw["state"], np.float64)[None])
                stats["actions"].update(commands.astype(np.float64))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if stats_path.exists():
            raise FileExistsError(
                "Refusing to replace existing norm stats; use --existing-norm or a new path"
            )
        stats_path.write_text(serialize_json({k: v.get_statistics() for k, v in stats.items()}) + "\n")
    processor = Processor(cfg, stats_path, quantile=options.get("quantile", True))
    manifest = {"format": "capo_demonstrations_v1", "data_spec": cfg.data_spec, "shards": [], "raw": [], "td_transitions": 0}
    manifest["transition_metadata"] = {
        k: getattr(cfg, k)
        for k in ("action_horizon", "execution_horizon", "action_dim", "robot_action_dim", "gamma")
    }
    pending = []
    transitions = []

    def flush():
        path = output / f"samples_{len(manifest['shards']):06d}.pt"
        torch.save({"samples": pending, "transitions": transitions}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        manifest["shards"].append({"file": path.name, "count": len(pending), "sha256": file_sha256(path)})
        pending.clear()
        transitions.clear()

    for path in files:
        manifest["raw"].append({"path": str(path), "sha256": file_sha256(path)})
        
        with np.load(path, allow_pickle=False) as source:
            td_data = {k: source[k] for k in source.files if k.startswith("next_image/") or k in
                       {"actions", "applied_actions", "next_state", "rewards", "terminated", "truncated"}}
        for index, (raw, commands) in enumerate(raw_samples(path, cfg)):
            mask = torch.zeros(cfg.action_horizon, cfg.action_dim, dtype=torch.bool)
            mask[: len(commands), : cfg.robot_action_dim] = True
            mask[:, list(cfg.inactive_action_dims)] = False
            pending.append(
                {"obs": processor.observation(raw), "actions": processor.actions(commands), "mask": mask}
            )
            transition = raw_transition(path, index, raw, commands, cfg, processor, archive=td_data)
            if transition is not None:
                transitions.append(transition)
                manifest["td_transitions"] += 1
            if len(pending) == 64:
                flush()
    if pending:
        flush()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cfg.save(output / "data_config.json")
    return cfg


class DemonstrationDataset:

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = local_path(cfg.demonstrations)
        self.manifest = load_mapping(self.root / "manifest.json")
        if self.manifest["format"] != "capo_demonstrations_v1":
            raise ValueError("Unsupported demonstration format")
        check_spec(cfg, self.manifest["data_spec"])
        self.identity = file_sha256(self.root / "manifest.json")
        self.entries, self.cache = [], OrderedDict()
        for shard in self.manifest["shards"]:
            path = local_path(self.root / shard["file"])
            if file_sha256(path) != shard["sha256"]:
                raise ValueError("Demonstration shard changed")
            self.entries.extend((path, i) for i in range(shard["count"]))
        if not self.entries:
            raise ValueError("Demonstrations are empty")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, offset = self.entries[index]
        if path not in self.cache:
            self.cache[path] = torch.load(path, weights_only=True)["samples"]
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        sample = self.cache[path][offset]
        if set(sample) != {"obs", "actions", "mask"}:
            raise ValueError("SFT samples must contain only obs/actions/mask")
        shape = (self.cfg.action_horizon, self.cfg.action_dim)
        if sample["actions"].shape != shape or sample["mask"].shape != shape:
            raise ValueError("Demonstration action shape differs from training config")
        if sample["mask"].dtype != torch.bool or not sample["mask"].any():
            raise ValueError("Demonstration needs a nonempty boolean action mask")
        if not torch.isfinite(sample["actions"]).all():
            raise ValueError("Nonfinite demonstration actions")
        return sample

    def sample(self, count, rng, start=0, end=None):
        indices = rng.choices(range(len(self)), k=count)
        return stack_trees([self[i] for i in indices[start:end]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "prepare"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--existing-norm", action="store_true", help="Reuse task stats computed with OpenPI")
    args = parser.parse_args()
    cfg, options = dataset_config(args.config)
    if args.stage == "prepare":
        prepare(cfg, options, existing_norm=args.existing_norm)
        print(local_path(options["output"]) / "data_config.json")
    else:
        from capo.robots.interface import create_adapter

        if args.episodes < 1:
            raise ValueError("episodes must be positive")
        robot = create_adapter(options["adapter"], cfg)
        try:
            print(f"Collected {collect_raw(cfg, options, robot, args.episodes)} supervised samples")
        finally:
            if hasattr(robot, "close"):
                robot.close()


if __name__ == "__main__":
    main()
