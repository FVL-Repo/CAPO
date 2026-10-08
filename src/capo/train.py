import argparse

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
        description="CAPO pi_05: offline replay or alternating online shard ingestion"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Continue optimizer/RNG/phase/reference state")
    parser.add_argument("--actor-init", help="Load a CAPO SFT actor, start fresh critics and optimizers")
    parser.add_argument("--stage", choices=("rl", "sft"), default="rl")
    parser.add_argument("--save-every", type=int, default=100)
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
        replay.refresh(local_path(cfg.replay))
    if not args.resume and not replay.records:
        raise ValueError("Replay is empty. Import demonstrations or collect data first.")
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

    def update():
        metrics = trainer.cvl_update() if trainer.phase in ("critic", "warmup") else trainer.actor_update()
        log_metrics(output / "metrics.jsonl", trainer, metrics)
        if args.save_every > 0 and trainer.phase_step % args.save_every == 0:
            trainer.save(output / "latest.pt")

    if trainer.phase == "sft":
        raise ValueError("Use --actor-init to start RL from SFT with fresh optimizers")

    if trainer.critic_step < cfg.critic_warmup_steps:
        if trainer.phase != "warmup":
            trainer.set_phase("warmup")
        while trainer.critic_step < cfg.critic_warmup_steps:
            update()
    if trainer.phase == "warmup":
        trainer.set_phase("idle")
        trainer.save(output / "latest.pt")
    while trainer.round < cfg.rounds:
        if trainer.phase == "idle":
            replay.refresh(local_path(cfg.replay))
            trainer.set_phase("critic")
        if trainer.phase == "critic":
            while trainer.phase_step < cfg.critic_steps_per_round:
                update()
            trainer.set_phase("actor")
            trainer.save(output / "latest.pt")
        if trainer.phase == "actor":
            while trainer.phase_step < cfg.actor_steps_per_round:
                update()
            trainer.round += 1
            trainer.set_phase("idle")
            trainer.save(output / "latest.pt")
    print(f"Finished. Checkpoint: {output / 'latest.pt'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if parallel.active():
            torch.distributed.destroy_process_group()
