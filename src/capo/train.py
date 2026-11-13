import argparse
import time

import torch

from capo import distributed as parallel
from capo.adjoint import cutoff_index
from capo.backbone import make_backbone
from capo.config import Config, local_path
from capo.replay import Replay
from capo.trainer import Trainer, log_metrics
from capo.checkpoints import load_actor_state


def main():
    parser = argparse.ArgumentParser(
        description="CAPO: one-time warmup, C / (C+A) updates and persistent reference snapshots"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Continue optimizer/RNG/phase/reference state")
    parser.add_argument("--actor-init", help="Load a CAPO SFT actor, start fresh critics and optimizers")
    parser.add_argument("--stage", choices=("rl", "sft"), default="rl")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--streaming", action="store_true", help="Wait for initial replay and ingest new shards while learning")
    parser.add_argument("--publish-dir")
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    if args.resume and args.actor_init:
        parser.error("--resume and --actor-init are mutually exclusive")
    cfg = Config.load(local_path(args.config))
    if cfg.backend == "pi05" and (cfg.data_spec is None or not cfg.data_spec.get("task")):
        raise ValueError(
            "pi05 training requires a bound data_spec; use the collector/importer's data_config.json"
        )
    if args.stage == "sft":
        if args.actor_init:
            raise ValueError("SFT starts from config.checkpoint; --actor-init is an RL option")
        from capo.sft import run

        run(cfg, resume=args.resume, save_every=args.save_every)
        return
    parallel.initialize(cfg)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    effective = cutoff_index(cfg.cutoff, cfg.flow_steps, cfg.cutoff_rule) / cfg.flow_steps
    print(
        f"backend={cfg.backend}; cutoff={cfg.cutoff}, effective cutoff={effective}; forward steps={cfg.flow_steps}",
        flush=True,
    )
    output = local_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    if parallel.rank() == 0:
        cfg.save(output / "config.json")
    replay = Replay(cfg)
    if not args.resume:
        replay.load_demonstrations()
        replay.refresh(local_path(cfg.replay))
    def stop_requested():
        import torch.distributed as dist
        stop = bool(args.stop_file and local_path(args.stop_file).exists()) if parallel.rank() == 0 else None
        if parallel.active():
            message = [stop]
            dist.broadcast_object_list(message, src=0)
            stop = message[0]
        return stop

    def ready():
        return all(not weight or replay.buffers[name]
                   for name, weight in zip(("demo", "exec", "inter"), cfg.replay_mixture))

    if not args.resume:
        while not ready():
            if stop_requested():
                raise RuntimeError("Stopped before initial replay became ready")
            if not args.streaming:
                raise ValueError("Import/collect data for each positive-weight replay buffer before training")
            if parallel.rank() == 0:
                print("Waiting for initial demo/exec/inter replay", flush=True)
            time.sleep(1)
            replay.refresh(local_path(cfg.replay))
    actor = make_backbone(cfg)
    if parallel.rank() == 0:
        cfg.save(output / "config.json")
    if args.actor_init:
        init = torch.load(local_path(args.actor_init), map_location="cpu", weights_only=True)
        load_actor_state(actor, init["actor"], init["config"])
    trainer = Trainer(cfg, actor, replay)
    torch.manual_seed(cfg.seed + parallel.rank())
    if args.resume:
        trainer.restore(local_path(args.resume))

    from capo.exchange import publish_actor
    from capo.schedule import validate_schedule_state

    validate_schedule_state(trainer)
    if trainer.actor_step > cfg.max_actor_steps:
        raise ValueError("Checkpoint exceeds requested actor budget")
    
    if args.publish_dir and parallel.rank() == 0:
        publish_actor(actor, cfg, local_path(args.publish_dir), trainer.actor_step)
    while trainer.actor_step < cfg.max_actor_steps:
        if stop_requested():
            break
        replay.refresh(local_path(cfg.replay))
        metrics = trainer.scheduled_update()
        log_metrics(output / "metrics.jsonl", trainer, metrics)
        optimizer_steps = trainer.critic_step + trainer.actor_step
        if (args.save_every > 0 and optimizer_steps % args.save_every == 0) or metrics.get("reference_refreshed"):
            trainer.save(output / "latest.pt")
        if (args.publish_dir and trainer.phase == "actor"
                and trainer.actor_step % cfg.policy_publish_interval == 0 and parallel.rank() == 0):
            publish_actor(actor, cfg, local_path(args.publish_dir), trainer.actor_step)
    trainer.set_phase("idle")
    trainer.save(output / "latest.pt")
    if args.publish_dir and parallel.rank() == 0:
        publish_actor(actor, cfg, local_path(args.publish_dir), trainer.actor_step)
    print(f"Finished. Checkpoint: {output / 'latest.pt'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if parallel.active():
            torch.distributed.destroy_process_group()
