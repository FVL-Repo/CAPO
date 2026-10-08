import argparse
import json

import torch

from capo.config import Config, local_path
from capo.replay import Replay, tensor_record
from capo.batching import stack_trees, tree_map
from capo.value import Critic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay", help="HELD-OUT replay shards, not the training replay")
    parser.add_argument("--training-replay", help="Check held-out episode/event disjointness")
    parser.add_argument("--continuations", help="JSONL with explicit frozen-policy continuation measurements")
    parser.add_argument("--normalizer", type=float, help="Explicit positive NMAE normalization Z")
    parser.add_argument("--policy-version", help="Restrict episode metrics to one immutable policy")
    parser.add_argument("--episodes", help="Collection episodes.jsonl")
    parser.add_argument("--last", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.last < 1:
        parser.error("--last must be positive")
    payload = torch.load(local_path(args.checkpoint), map_location="cpu", weights_only=True)
    if payload.get("critic_architecture") != Critic.architecture:
        raise ValueError(
            "Incompatible critic checkpoint: evaluation requires the 224-pixel direct-Q architecture."
        )
    cfg = Config(**payload["config"]).validate()
    cfg.device = args.device
    report = {}
    if args.replay:
        replay = Replay(cfg)
        replay.refresh(local_path(args.replay))
        if args.training_replay:
            from capo.evaluation import check_heldout

            training = Replay(cfg)
            training.refresh(local_path(args.training_replay))
            report.update(check_heldout(training.records, replay.records))
        critic = Critic(cfg).to(cfg.device).eval()
        critic.load_state_dict(payload["critic"])
        agreements, margins, head_satisfied = [], [], []
        with torch.no_grad():
            for index in replay.pairs:
                batch = stack_trees([tensor_record(replay.records[index], cfg)])
                batch = tree_map(lambda t: t.to(cfg.device), batch)
                mask = batch["mask"][:, : cfg.execution_horizon, : cfg.robot_action_dim]
                features = critic.encode(batch["obs"])
                good = critic.from_features(
                    features, batch["actions"][:, : cfg.execution_horizon, : cfg.robot_action_dim], mask
                )
                bad = critic.from_features(
                    features,
                    batch["rejected_actions"][:, : cfg.execution_horizon, : cfg.robot_action_dim],
                    mask,
                )
                difference = good.min(-1).values - bad.min(-1).values
                agreements.append((difference > 0).float().item())
                margins.append(difference.item())
                head_satisfied.append(((good - bad) >= cfg.margin).float().mean().item())
        report.update(
            heldout_pairs=len(agreements),
            preference_accuracy=sum(agreements) / len(agreements) if agreements else None,
            mean_min_q_margin=sum(margins) / len(margins) if margins else None,
            per_head_margin_satisfied=sum(head_satisfied) / len(head_satisfied) if head_satisfied else None,
        )
    if args.continuations:
        from capo.evaluation import continuation_metrics

        if args.normalizer is None:
            parser.error("--continuations requires --normalizer Z")
        rows = [
            json.loads(line)
            for line in local_path(args.continuations).read_text().splitlines()
            if line.strip()
        ]
        report.update(continuation_metrics(rows, args.normalizer))
    if args.episodes:
        episodes = [
            json.loads(line) for line in local_path(args.episodes).read_text().splitlines() if line.strip()
        ]
        if args.policy_version:
            episodes = [e for e in episodes if e["policy_version"] == args.policy_version]
        episodes = episodes[-args.last :]
        steps = sum(e["primitive_steps"] for e in episodes)
        report.update(
            episodes=len(episodes),
            autonomous_success_rate=sum(e["autonomous_success"] for e in episodes) / len(episodes)
            if episodes
            else None,
            primitive_intervention_ratio=sum(e["human_steps"] for e in episodes) / steps if steps else None,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
