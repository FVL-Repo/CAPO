import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from capo.checkpoints import check_actor_config, load_actor_state

from capo import distributed as parallel
from capo.adjoint import build_targets, flow_matching_loss, rollout, squared_norm_loss
from capo.batching import slice_batch, tree_map
from capo.demonstrations import DemonstrationDataset
from capo.imitation import TRAINING_OBJECTIVE
from capo.schedule import TRAINING_SCHEDULE, next_stage, validate_schedule_state
from capo.value import Critic, polyak_update, ranking_loss, td_target


def learning_rate(step, peak, minimum, warmup, total):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * fraction))


class Trainer:
    def __init__(self, cfg, actor, replay):
        from capo.identity import canonical

        if cfg.train_vlm:
            raise ValueError("Online RL freezes the VLM; export full SFT and use its capo_config.json")
        if canonical(cfg.data_spec) != canonical(replay.cfg.data_spec):
            raise ValueError("Trainer and Replay data specifications differ")
        if cfg.backend == "pi05":
            if cfg.base_sha256 and cfg.base_sha256 != actor.cfg.base_sha256:
                raise ValueError("Trainer/actor base mismatch")
            cfg.base_sha256 = actor.cfg.base_sha256
        self.cfg, self.actor, self.replay = cfg, actor, replay
        self.demonstrations = DemonstrationDataset(cfg) if cfg.demonstrations else None
        if cfg.batch_size % parallel.world_size():
            raise ValueError("Global batch_size must be divisible by world size")
        self.local_batch_size = cfg.batch_size // parallel.world_size()
        self.critic = Critic(cfg).to(cfg.device)
        parallel.broadcast_model(actor)
        parallel.broadcast_model(self.critic)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_parameters = [p for p in actor.parameters() if p.requires_grad]
        options = dict(betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
        self.actor_optimizer = torch.optim.AdamW(self.actor_parameters, lr=cfg.actor_lr, **options)
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.critic.parameters() if p.requires_grad], lr=cfg.critic_lr, **options
        )
        self.actor_step = self.critic_step = self.sft_step = 0
        self.schedule_stage = "warmup" if cfg.critic_warmup_steps else "critic_only"
        self.reference_actor_step = 0
        self.reference_refresh_count = 0
        self.reference = None
        self.phase = "idle"
        self.phase_step = 0

    def sample_batch(self, view="cvl"):

        start = parallel.rank() * self.local_batch_size
        return self.replay.sample(
            self.cfg.batch_size, view, cfg=self.cfg, start=start, end=start + self.local_batch_size
        )

    def project(self, actions):
        result = actions[:, : self.cfg.execution_horizon, : self.cfg.robot_action_dim]
        if self.cfg.inactive_action_dims:
            result = result.clone()
            result[..., list(self.cfg.inactive_action_dims)] = 0
        return result

    def noise(self, count):
        c = self.cfg
        return torch.randn(count, c.action_horizon, c.action_dim, device=c.device)

    def sample_actions(self, context, count):
        return rollout(
            lambda x, t: self.actor.velocity(context, x, t), self.noise(count), self.cfg.flow_steps
        )[-1]

    def set_phase(self, phase):
        self.phase, self.phase_step = phase, 0
        self.critic.requires_grad_(phase in ("critic", "warmup"))
        self.critic.zero_grad(set_to_none=True)
        self.actor.zero_grad(set_to_none=True)
        if phase == "actor" and self.reference is None:
            self.refresh_reference()

    def refresh_reference(self):
        if self.reference is not None:
            self.reference_refresh_count += 1
        self.reference = self.actor.reference()
        self.reference_actor_step = self.actor_step

    def scheduled_update(self):
        """One optimizer update; the joint slot is critic then actor, resumable between them."""
        validate_schedule_state(self)
        stage = self.schedule_stage
        if self.actor_step >= self.cfg.max_actor_steps:
            raise RuntimeError("Actor update budget already reached")
        phase = "warmup" if stage == "warmup" else ("actor" if stage == "joint_actor" else "critic")
        self.set_phase(phase)
        metrics = self.actor_update() if phase == "actor" else self.cvl_update()
        self.schedule_stage = next_stage(stage, self.critic_step, self.cfg.critic_warmup_steps)
        validate_schedule_state(self)
        return {**metrics, "schedule_slot": stage}

    def finish_optimizer(self, optimizer, params):
        params = list(params)
        parallel.average_gradients(params)
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite gradients: refusing to update weights")
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.phase_step += 1

    def cvl_update(self):
        if self.phase not in ("critic", "warmup"):
            raise RuntimeError("Call set_phase('critic') first")
        c = self.cfg
        batch = self.sample_batch()
        optimizer = self.critic_optimizer
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(
            self.critic_step, c.critic_lr, c.critic_min_lr, c.critic_lr_warmup, c.critic_schedule_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = {"td": 0.0, "rank": 0.0, "critic_lr": lr}
        for index, name in enumerate(("demo", "exec", "inter")):
            selected = batch["buffer_id"] == index
            metrics[f"td_samples_{name}"] = int(selected.sum()) * parallel.world_size()
            metrics[f"td_corrections_{name}"] = int((selected & (batch["source_id"] == 1)).sum()) * parallel.world_size()
        metrics["ranking_pairs_used"] = int(batch["ranking_mask"].sum()) * parallel.world_size() if c.rank_weight else 0
        for start in range(0, self.local_batch_size, c.microbatch_size):
            end = min(self.local_batch_size, start + c.microbatch_size)
            b = slice_batch(batch, start, end, c.device)
            fraction = (end - start) / self.local_batch_size
            with torch.no_grad():
                ctx = self.actor.prepare(b["next_obs"])
                features = self.target.encode(b["next_obs"])
                values = []
                for _ in range(c.td_samples):
                    actions = self.project(self.sample_actions(ctx, end - start))
                    values.append(self.target.from_features(features, actions))
                y = td_target(b["reward"], b["terminated"], b["duration"], torch.stack(values, 1), c.gamma)
                del ctx, features, values
            # All demo/exec samples, plus the executed corrective side of inter.
            # Rejected inter proposals have no observed reward/successor and never get TD.
            q = self.critic(b["obs"], self.project(b["actions"]), self.project(b["mask"]))
            td = (q - y[:, None]).square().mean()
            (td * fraction).backward()
            metrics["td"] += td.detach().item() * fraction
            pair_mask = b["ranking_mask"]
            if c.rank_weight and pair_mask.any():
                p = tree_map(lambda t: t[pair_mask], b)
                features = self.critic.encode(p["obs"])
                mask = self.project(p["mask"])
                human = self.critic.from_features(features, self.project(p["actions"]), mask)
                rejected = self.critic.from_features(features, self.project(p["rejected_actions"]), mask)
                rank = ranking_loss(human, rejected, c.margin)
                # Mean over the GLOBAL inter subset, even when ranks/microbatches
                # contain different pair counts (including zero).
                rank_fraction = p["ranking_weight"].sum() / self.local_batch_size
                (c.rank_weight * rank_fraction * rank).backward()
                metrics["rank"] += rank.detach().item() * rank_fraction.item()
        self.finish_optimizer(optimizer, self.critic.parameters())
        polyak_update(self.target, self.critic, c.target_retention)
        self.critic_step += 1
        return parallel.mean_metrics(metrics, c.device)

    def actor_update(self):
        if self.phase != "actor":
            raise RuntimeError("Actor update called in the wrong phase")
        c = self.cfg
        batch = self.sample_batch("actor")
        optimizer = self.actor_optimizer
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(
            self.actor_step, c.actor_lr, c.actor_min_lr, c.actor_lr_warmup, c.actor_schedule_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = {"apo": 0.0, "bc": 0.0, "actor_lr": lr, "reference_actor_step_used": self.reference_actor_step}
        for index, name in enumerate(("demo", "exec", "inter")):
            selected = batch["buffer_id"] == index
            metrics[f"actor_samples_{name}"] = int(selected.sum()) * parallel.world_size()
            metrics[f"actor_corrections_{name}"] = int((selected & (batch["source_id"] == 1)).sum()) * parallel.world_size()
            metrics[f"bc_samples_{name}"] = int((selected & batch["bc_mask"]).sum()) * parallel.world_size() if c.bc_weight else 0
            metrics[f"apo_samples_{name}"] = int((selected & batch["apo_mask"]).sum()) * parallel.world_size()
        metrics["bc_autonomous_used"] = int((batch["bc_mask"] & (batch["source_id"] == 2)).sum()) * parallel.world_size() if c.bc_weight else 0
        for start in range(0, self.local_batch_size, c.microbatch_size):
            end = min(self.local_batch_size, start + c.microbatch_size)
            b = slice_batch(batch, start, end, c.device)
            # One shared rounded 40/40/20 draw. APO uses exec states; BC uses
            # only samples drawn through demo (experts + human interventions).
            if b["apo_mask"].any():
                a = tree_map(lambda t: t[b["apo_mask"]], b)
                fraction = (a["apo_weight"].sum() / self.local_batch_size).item()
                ctx = self.actor.prepare(a["obs"])
                with torch.no_grad():
                    features = self.critic.encode(a["obs"]).detach()

                def value(a, features=features):
                    return self.critic.from_features(features, self.project(a)).min(-1).values

                def velocity(x, t, ctx=ctx):
                    return self.actor.velocity(ctx, x, t)

                def reference(x, t, ctx=ctx):
                    return self.reference.velocity(ctx, x, t)

                states = rollout(velocity, self.noise(len(a["actions"])), c.flow_steps)
                targets = build_targets(
                    reference,
                    states,
                    value,
                    eta=c.eta,
                    alpha_q=c.alpha_q,
                    cutoff=c.cutoff,
                    cutoff_rule=c.cutoff_rule,
                )
                indices = list(targets.velocities)

                selected = torch.randint(len(indices), (c.apo_time_samples, len(a["actions"])), device=c.device)
                grid = torch.tensor(indices, device=c.device)
                target_stack = torch.stack([targets.velocities[i] for i in indices])
                rows = torch.arange(len(a["actions"]), device=c.device)
                for selections in selected:
                    steps = grid[selections]
                    x = states[steps, rows].detach()
                    target = target_stack[selections, rows].detach()
                    loss = squared_norm_loss(velocity(x, steps.to(x.dtype) / c.flow_steps), target)
                    (loss * fraction / c.apo_time_samples).backward()
                    metrics["apo"] += loss.detach().item() * fraction / c.apo_time_samples
                del ctx, features, targets, states, target_stack, value, velocity, reference
            if c.bc_weight and b["bc_mask"].any():
                d = tree_map(lambda t: t[b["bc_mask"]], b)
                demo_ctx = self.actor.prepare(d["obs"])
                bc = flow_matching_loss(
                    lambda x, t: self.actor.velocity(demo_ctx, x, t), d["actions"], d["mask"]
                )
                fraction = (d["bc_weight"].sum() / self.local_batch_size).item()
                (bc * fraction * c.bc_weight).backward()
                metrics["bc"] += bc.detach().item() * fraction
                del demo_ctx
        self.finish_optimizer(optimizer, self.actor_parameters)
        self.actor_step += 1
        refreshed = self.actor_step - self.reference_actor_step == c.reference_refresh_interval
        if refreshed:
            self.refresh_reference()
        metrics.update(reference_refreshed=int(refreshed), reference_actor_step=self.reference_actor_step,
                       actor_steps_since_reference=self.actor_step - self.reference_actor_step)
        return parallel.mean_metrics(metrics, c.device)

    def save(self, path):

        rank_rng = parallel.gather_objects(
            {
                "torch_rng": torch.get_rng_state(),
                "replay_rng": self.replay.rng.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(self.cfg.device).cpu()
                if str(self.cfg.device).startswith("cuda")
                else None,
            }
        )
        if parallel.rank() != 0:
            return
        payload = {
            "training_objective": TRAINING_OBJECTIVE,
            "training_schedule": TRAINING_SCHEDULE,
            "schedule_stage": self.schedule_stage,
            "reference_actor_step": self.reference_actor_step,
            "reference_refresh_count": self.reference_refresh_count,
            "demonstration_identity": (self.demonstrations.identity if self.demonstrations else None),
            "replay_snapshot": self.replay.snapshot(),
            "world_size": parallel.world_size(),
            "rank_rng": rank_rng,
            "format_version": 1,
            "critic_architecture": self.critic.architecture,
            "config": asdict(self.cfg),
            "actor": self.actor.trainable_state(),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_step": self.actor_step,
            "critic_step": self.critic_step,
            "sft_step": self.sft_step,
            "phase": self.phase,
            "phase_step": self.phase_step,
            "torch_rng": torch.get_rng_state(),
            "replay_rng": self.replay.rng.getstate(),
            "cuda_rng": [],
            "reference": None,
        }
        if self.reference is not None:
            keys = set(payload["actor"])
            payload["reference"] = {
                k: v.cpu().clone() for k, v in self.reference.state_dict().items() if k in keys
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(path)

    def restore(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("training_schedule") != TRAINING_SCHEDULE:
            raise ValueError("RL checkpoint uses a different update schedule; use --actor-init and a new output directory")
        if payload.get("training_objective") != TRAINING_OBJECTIVE:
            raise ValueError("RL checkpoint uses a different BC objective; use --actor-init and a new output directory")
        if payload.get("world_size", 1) != parallel.world_size():
            raise ValueError("Exact resume requires the same world size; use --actor-init to change it")
        if payload["format_version"] != 1:
            raise ValueError("Unsupported checkpoint")
        if payload.get("critic_architecture") != self.critic.architecture:
            raise ValueError(
                "Legacy/incompatible critic checkpoint. Use --actor-init to keep the actor and train a fresh critic."
            )
        check_actor_config(self.actor, payload["config"])

        ignore = {"output", "device", "max_actor_steps", "replay", "sft_steps"}
        for k, v in asdict(self.cfg).items():
            old = payload["config"].get(k, v)
            if isinstance(v, (tuple, list)):
                old, v = tuple(old), tuple(v)
            if k not in ignore and v != old:
                raise ValueError(f"Resume config mismatch for {k}")
        if "replay_snapshot" not in payload:
            raise ValueError("Legacy checkpoint lacks replay snapshot; use --actor-init")
        if payload.get("demonstration_identity") != (self.demonstrations.identity if self.demonstrations else None):
            raise ValueError("Offline demonstrations changed since RL checkpoint")
        self.replay.restore_snapshot(payload["replay_snapshot"])
        load_actor_state(self.actor, payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target.load_state_dict(payload["target"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.reference = None
        if payload["reference"] is not None:
            self.reference = self.actor.reference()
            self.reference.load_state_dict(payload["reference"], strict=False)
        for k in ("actor_step", "critic_step", "sft_step", "schedule_stage",
                  "reference_actor_step", "reference_refresh_count"):
            setattr(self, k, payload[k])
        self.set_phase(payload["phase"])
        self.phase_step = payload["phase_step"]
        validate_schedule_state(self)
        if self.actor_step and payload["reference"] is None:
            raise ValueError("Checkpoint is missing the persistent reference snapshot")
        if (self.reference_actor_step != (self.actor_step // self.cfg.reference_refresh_interval) * self.cfg.reference_refresh_interval
                or self.reference_refresh_count != self.actor_step // self.cfg.reference_refresh_interval):
            raise ValueError("Invalid checkpoint reference counter")
        torch.set_rng_state(payload["torch_rng"].cpu())
        self.replay.rng.setstate(payload["replay_rng"])
        if torch.cuda.is_available() and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
        if "rank_rng" in payload:
            state = payload["rank_rng"][parallel.rank()]
            torch.set_rng_state(state["torch_rng"].cpu())
            self.replay.rng.setstate(state["replay_rng"])
            if state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.cfg.device)


def log_metrics(path, trainer, metrics):
    if parallel.rank() != 0:
        return
    record = {
        "schedule_stage_next": trainer.schedule_stage,
        "reference_actor_step": trainer.reference_actor_step,
        "reference_refresh_count": trainer.reference_refresh_count,
        "actor_steps_since_reference": trainer.actor_step - trainer.reference_actor_step,
        "phase": trainer.phase,
        "phase_step": trainer.phase_step,
        "critic_step": trainer.critic_step,
        "actor_step": trainer.actor_step,
        "replay_transitions": len(trainer.replay.records),
        "replay_pairs": len(trainer.replay.pairs),
        **{f"buffer_{name}": len(indices) for name, indices in trainer.replay.buffers.items()},
        **metrics,
    }
    with Path(path).open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
