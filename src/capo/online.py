import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from capo.config import Config, ROOT, local_path
from capo.identity import file_sha256, canonical
from capo.processing import Processor
from capo.replay import Replay


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def publish(checkpoint, output):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    artifact = {"format_version": 1, "config": payload["config"], "actor": payload["actor"]}
    path = output / (checkpoint.parent.name + "_actor.pt")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite published policy {path}")
    temporary = path.with_suffix(".tmp")
    torch.save(artifact, temporary)
    temporary.replace(path)
    digest = file_sha256(path)
    atomic_json(output / "policy.json", {"checkpoint": str(path), "sha256": digest})
    return str(path), digest


def training_complete(payload, config):
    return (
        payload["phase"] == "idle"
        and payload["round"] == config.rounds
        and payload["actor_step"] == config.rounds * config.actor_steps_per_round
        and payload["critic_step"]
        == config.critic_warmup_steps + config.rounds * config.critic_steps_per_round
    )


def run(
    cfg,
    *,
    norm_stats,
    adapter,
    actor_init=None,
    reward_config=None,
    episodes_per_round=1,
    min_new_records=1,
    train_processes=1,
):
    output = local_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    if episodes_per_round < 1 or min_new_records < 1 or train_processes < 1:
        raise ValueError("Invalid online collection/training budget")
    processor = Processor(
        cfg, norm_stats, quantile=(cfg.data_spec or {}).get("normalization_mode", "quantile") == "quantile"
    )
    del processor
    if cfg.backend == "pi05":
        digest = file_sha256(local_path(cfg.checkpoint))
        if cfg.base_sha256 and cfg.base_sha256 != digest:
            raise ValueError("Base checkpoint changed")
        cfg.base_sha256 = digest
    reward_identity = None
    if reward_config:
        from capo.reward import load_reward_config

        reward = load_reward_config(reward_config)
        reward_identity = {
            "config": reward,
            "sha256": file_sha256(local_path(reward["checkpoint"])) if reward.get("enabled") else None,
        }
    identity = {
        "config": cfg.__dict__,
        "reward": reward_identity,
        "norm_stats": file_sha256(local_path(norm_stats)),
        "adapter": adapter,
        "train_processes": train_processes,
        "episodes_per_round": episodes_per_round,
        "min_new_records": min_new_records,
        "actor_init": str(local_path(actor_init)) if actor_init else None,
        "actor_init_sha256": file_sha256(local_path(actor_init)) if actor_init else None,
    }
    identity = json.loads(canonical(identity))
    state_path = output / "online_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Online run configuration changed; create a new output directory")
        if state.get("policy") and file_sha256(state["policy"]) != state["policy_sha256"]:
            raise ValueError("Published policy changed")
    else:
        if actor_init:
            initial = local_path(actor_init)
            from capo.checkpoints import check_actor_config
            from types import SimpleNamespace

            payload = torch.load(initial, map_location="cpu", weights_only=True)
            check_actor_config(SimpleNamespace(cfg=cfg), payload["config"])
            state = {
                "identity": identity,
                "phase": "publish",
                "round": 0,
                "training_checkpoint": str(initial),
            }
        else:
            state = {"identity": identity, "phase": "collect", "round": 0, "policy": None}
        atomic_json(state_path, state)

    def invoke(command, config, stage):
        path = output / f"{stage}_config.json"
        config.save(path)
        with (output / "online.log").open("a") as log:
            subprocess.run(
                command + ["--config", str(path)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def collect(config):
        command = [
            sys.executable,
            "-m",
            "capo.collect",
            "--norm-stats",
            str(norm_stats),
            "--adapter",
            adapter,
            "--episodes",
            str(episodes_per_round),
        ]
        if (cfg.data_spec or {}).get("normalization_mode") == "mean_std":
            command.append("--mean-std")
        if state.get("policy"):
            command += ["--checkpoint", state["policy"]]
        if reward_config:
            command += ["--reward-config", str(reward_config)]
        invoke(command, config, "collect")

    def train(config):
        command = [sys.executable, "-m"]
        if train_processes > 1:
            command += ["torch.distributed.run", "--standalone", f"--nproc-per-node={train_processes}", "-m"]
        command += ["capo.train", "--save-every", "1"]
        target = local_path(config.output) / "latest.pt"
        if target.exists():
            payload = torch.load(target, map_location="cpu", weights_only=True)
            if training_complete(payload, config):
                return target
            command += ["--resume", str(target)]
        elif state.get("training_checkpoint"):
            command += ["--actor-init" if state["round"] == 0 else "--resume", state["training_checkpoint"]]
        invoke(command, config, "train")
        payload = torch.load(target, map_location="cpu", weights_only=True)
        if not training_complete(payload, config):
            raise RuntimeError("Training did not reach the requested update count")
        return target

    while state["round"] < cfg.rounds:
        phase = state["phase"]
        if phase == "publish":
            target = Path(state["training_checkpoint"])
            path = output / (target.parent.name + "_actor.pt")
            if path.exists():
                old = torch.load(path, map_location="cpu", weights_only=True)
                current = torch.load(target, map_location="cpu", weights_only=True)
                if canonical(old["config"]) != canonical(current["config"]) or any(
                    not torch.equal(v, current["actor"][k]) for k, v in old["actor"].items()
                ):
                    raise ValueError("Existing publication differs from completed training")
                policy, digest = str(path), file_sha256(path)
            else:
                policy, digest = publish(target, output)
            state.update(policy=policy, policy_sha256=digest, phase="collect")
        elif phase == "collect":
            before = Replay(cfg)
            before.refresh(local_path(cfg.replay))
            count = sum(1 for _ in before.manifest)
            collect(cfg)
            after = Replay(cfg)
            after.refresh(local_path(cfg.replay))
            new = after.loaded - before.loaded
            new_records = sum(len(torch.load(p, weights_only=True)["records"]) for p in new)
            if new_records < min_new_records:
                raise RuntimeError(
                    f"Only {new_records} new valid records; minimum={min_new_records}; training not started"
                )
            state.update(
                phase="train", collected_shards=len(after.manifest) - count, collected_records=new_records
            )
        elif phase == "train":
            round_cfg = replace(
                cfg, rounds=state["round"] + 1, output=str(output / f"round_{state['round'] + 1:04d}")
            )
            target = train(round_cfg)
            state.update(training_checkpoint=str(target), round=state["round"] + 1, phase="publish")
        else:
            raise ValueError(f"Unknown online phase {phase}")
        atomic_json(state_path, state)
    if state["phase"] == "publish":
        target = Path(state["training_checkpoint"])
        path = output / (target.parent.name + "_actor.pt")
        if not path.exists():
            policy, digest = publish(target, output)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            current = torch.load(target, map_location="cpu", weights_only=True)
            if canonical(saved["config"]) != canonical(current["config"]) or any(
                not torch.equal(v, current["actor"][k]) for k, v in saved["actor"].items()
            ):
                raise ValueError("Final publication differs from completed training")
            policy, digest = str(path), file_sha256(path)
        state.update(policy=policy, policy_sha256=digest, phase="complete")
        atomic_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--reward-config")
    parser.add_argument(
        "--actor-init", help="Offline SFT delta; omit only for an external full OpenPI SFT base"
    )
    parser.add_argument("--episodes-per-round", type=int, default=1)
    parser.add_argument("--min-new-records", type=int, default=1)
    parser.add_argument("--train-processes", type=int, default=1)
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    if args.adapter == "capo.robots.flexiv_client:create":
        from capo.identity import bind_robot
        from capo.robots.flexiv_config import load_config

        bind_robot(
            cfg,
            load_config(local_path(os.environ.get("CAPO_FLEXIV_CONFIG", "configs/robots/flexiv.yaml"))),
        )
    print(
        json.dumps(
            run(
                cfg,
                norm_stats=args.norm_stats,
                adapter=args.adapter,
                actor_init=args.actor_init,
                reward_config=args.reward_config,
                episodes_per_round=args.episodes_per_round,
                min_new_records=args.min_new_records,
                train_processes=args.train_processes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
