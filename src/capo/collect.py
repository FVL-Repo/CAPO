import argparse
import uuid

import numpy as np

from capo.config import Config, local_path
from capo.policy import Policy
from capo.processing import Processor
from capo.replay import save_shard


from capo.robots.interface import create_adapter


def make_record(cfg, processor, obs, proposal, result, *, human, aligned, demonstration=False):
    actions = np.asarray(result.get("commands", result["actions"]), np.float32)
    rewards = np.asarray(result["rewards"], np.float32)
    duration = len(actions)
    if not 1 <= duration <= cfg.execution_horizon or rewards.shape != (duration,):
        raise ValueError("execute must report one reward per actually executed action")
    human_steps = int(result["human_steps"])
    if not 0 <= human_steps <= duration:
        raise ValueError("Invalid human_steps")
    if human_steps not in (0, duration):
        raise ValueError("Mixed-control chunks must be split at takeover boundaries")
    if bool(human_steps) != bool(human):
        raise ValueError("Reported control source changed; split/recollect the transition")
    record = {
        "obs": processor.observation(obs),
        "next_obs": processor.observation(result["next_obs"]),
        "actions": processor.actions(actions),
        "applied_actions": processor.actions(result["actions"]),
        "reward": float((cfg.gamma ** np.arange(duration) * rewards).sum()),
        "terminated": bool(result["terminated"]),
        "truncated": bool(result["truncated"]),
        "duration": duration,
        "source": "demonstration" if demonstration else ("correction" if human else "autonomous"),
    }
    record["metadata"] = {
        "norm_stats_sha256": processor.spec["norm_stats_sha256"],
        "normalization_mode": processor.spec["normalization_mode"],
        "task": str(obs["prompt"]),
        "data_spec": processor.spec,
    }
    for key in ("reward_probability", "reward_feedback", "manual_success"):
        if key in result:
            record["metadata"][key] = result[key]
    if "robot_metadata" in obs:
        record["metadata"].update(
            {
                **obs["robot_metadata"],
                "decision_id": obs.get("decision_id"),
                "success": bool(result["success"]),
                "safety_abort": bool(result["safety_abort"]),
            }
        )
    if human and aligned and not demonstration:
        if len(proposal) < duration:
            raise ValueError("Rejected proposal must cover the executed horizon")
        record.update(rejected_actions=processor.actions(proposal[:duration]), pair_aligned=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--mean-std", action="store_true", help="Match mean/std-normalized replay")
    parser.add_argument("--adapter", required=True, help="Project-local Python module:factory, factory(cfg)")
    parser.add_argument(
        "--reward-config", help="Optional image success classifier, otherwise adapter/manual reward"
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--demonstration", action="store_true")
    args = parser.parse_args()
    cfg = Config.load(local_path(args.config))
    if args.adapter == "capo.robots.flexiv_client:create":
        import os
        from capo.identity import bind_robot
        from capo.robots.flexiv_config import load_config

        bind_robot(
            cfg,
            load_config(local_path(os.environ.get("CAPO_FLEXIV_CONFIG", "configs/robots/flexiv.yaml"))),
        )
    processor = Processor(cfg, args.norm_stats, quantile=not args.mean_std)

    policy = None if args.demonstration else Policy(cfg, args.checkpoint, processor)
    robot = create_adapter(args.adapter, cfg)
    if hasattr(robot, "demonstration"):
        robot.demonstration = args.demonstration
    try:
        if args.reward_config or args.adapter == "capo.robots.flexiv_client:create":
            from capo.reward import wrap_reward

            robot = wrap_reward(robot, args.reward_config)
        from capo.identity import file_sha256

        collect_episodes(
            cfg,
            processor,
            robot,
            policy,
            episodes=args.episodes,
            demonstration=args.demonstration,
            policy_version=("sha256:" + file_sha256(local_path(args.checkpoint)))
            if args.checkpoint
            else "demonstration",
        )
    finally:
        if hasattr(robot, "close"):
            robot.close()


def collect_episodes(cfg, processor, robot, policy, *, episodes=10, demonstration=False, policy_version=None):
    import json
    from .robots.flexiv_rpc import pack

    directory = local_path(cfg.replay)
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / "staging"
    staging.mkdir(exist_ok=True)
    cfg.save(directory / "data_config.json")
    reports = []

    def durable(path, data):
        import os

        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(pack(data))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    for episode in range(episodes):
        obs = robot.reset()
        episode_id = obs.get("robot_metadata", {}).get("episode_id", uuid.uuid4().hex)
        steps = human_count = 0
        pending_record = pending_id = None
        success = safety_abort = truncated = False

        def publish_pending():
            nonlocal pending_record, pending_id
            if pending_record is not None:
                save_shard(directory / f"{pending_id}.pt", [pending_record], cfg)
                pending_record = pending_id = None

        try:
            while True:
                proposal = (
                    np.zeros((cfg.execution_horizon, cfg.robot_action_dim), np.float32)
                    if policy is None
                    else policy.infer(obs)["actions"][: cfg.execution_horizon]
                )
                selected, human, aligned = robot.choose_action(obs, proposal.copy())
                if demonstration and not human:
                    raise ValueError("Demonstrations must be explicitly human-controlled")
                decision_id = obs.get("decision_id", uuid.uuid4().hex)
                staged = {
                    "obs": obs,
                    "proposal": proposal,
                    "selected": selected,
                    "human": human,
                    "aligned": aligned,
                    "demonstration": demonstration,
                    "policy_version": policy_version,
                    "episode_id": episode_id,
                }
                stage_path = staging / (decision_id + ".msgpack")
                durable(stage_path, staged)
                result = robot.execute(selected)
                durable(stage_path, {**staged, "result": result})
                if result.get("quarantined") or result.get("next_obs") is None:
                    quarantine = directory / "quarantine"
                    quarantine.mkdir(exist_ok=True)
                    stage_path.replace(quarantine / stage_path.name)
                    safety_abort = truncated = True
                    if pending_record is not None:
                        pending_record["truncated"] = True
                    publish_pending()
                    break
                if result.get("boundary"):
                    if pending_record is not None:
                        override = result.get("reward_override")
                        if override is not None:
                            old = pending_record["metadata"]["last_primitive_reward"]
                            pending_record["reward"] += cfg.gamma ** (pending_record["duration"] - 1) * (
                                override - old
                            )
                        pending_record["terminated"] = result["terminated"]
                        pending_record["metadata"]["success"] = result["success"]
                        for key in ("reward_probability", "reward_feedback", "manual_success"):
                            if key in result:
                                pending_record["metadata"][key] = result[key]

                        pending_record["next_obs"] = processor.observation(result["next_obs"])
                    obs = result["next_obs"]
                    if result["terminated"]:
                        success = result["success"]
                        publish_pending()
                        break
                    continue
                if result.get("retry"):
                    obs = result["next_obs"]
                    continue
                publish_pending()
                record = make_record(
                    cfg,
                    processor,
                    obs,
                    proposal,
                    result,
                    human=human,
                    aligned=aligned,
                    demonstration=demonstration,
                )
                record["metadata"].update(
                    policy_version=policy_version,
                    episode_id=episode_id,
                    decision_id=decision_id,
                    transition_index=steps,
                    last_primitive_reward=float(result["rewards"][-1]),
                )
                pending_record, pending_id = record, decision_id
                steps += record["duration"]
                human_count += int(result["human_steps"])
                obs = result["next_obs"]
                if record["terminated"] or record["truncated"]:
                    success, safety_abort, truncated = (
                        result["success"],
                        result["safety_abort"],
                        record["truncated"],
                    )
                    publish_pending()
                    break
        except BaseException:
            if pending_record is not None:
                pending_record["truncated"] = True
                pending_record["metadata"]["collection_aborted"] = True
                publish_pending()
            raise
        report = {
            "episode": episode,
            "episode_id": episode_id,
            "policy_version": policy_version,
            "primitive_steps": steps,
            "human_steps": human_count,
            "intervention_ratio": human_count / max(steps, 1),
            "autonomous_success": bool(success and not human_count and not truncated and not safety_abort),
            "safety_abort": bool(safety_abort),
            "truncated": bool(truncated),
        }
        print(json.dumps(report), flush=True)
        with (directory / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        reports.append(report)
    return reports


if __name__ == "__main__":
    main()
