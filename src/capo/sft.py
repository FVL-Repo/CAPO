import argparse
from dataclasses import asdict
import json
import random

import torch

from capo import distributed as parallel
from capo.adjoint import flow_matching_loss
from capo.backbone import make_backbone
from capo.batching import slice_batch
from capo.checkpoints import load_actor_state
from capo.config import Config, local_path
from capo.demonstrations import DemonstrationDataset


class SFTTrainer:
    def __init__(self, cfg, actor, dataset):
        self.cfg, self.actor, self.dataset = cfg, actor, dataset
        self.rng = random.Random(cfg.seed)
        self.step = 0
        self.params = [p for p in actor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.params, lr=cfg.sft_lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay
        )
        parallel.broadcast_model(actor)

    def update(self):
        c = self.cfg
        size = c.batch_size // parallel.world_size()
        start = parallel.rank() * size
        batch = self.dataset.sample(c.batch_size, self.rng, start, start + size)
        lr = c.sft_lr * min(1, (self.step + 1) / max(1, c.sft_lr_warmup))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for start in range(0, size, c.microbatch_size):
            end = min(size, start + c.microbatch_size)
            b = slice_batch(batch, start, end, c.device)
            ctx = self.actor.prepare(b["obs"])
            loss = flow_matching_loss(lambda x, t: self.actor.velocity(ctx, x, t), b["actions"], b["mask"])
            (loss * ((end - start) / size)).backward()
            loss_value += loss.detach().item() * (end - start) / size
        parallel.average_gradients(self.params)
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.params):
            raise FloatingPointError("Nonfinite SFT gradients")
        if c.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, c.grad_clip)
        self.optimizer.step()
        self.step += 1
        return parallel.mean_metrics({"bc": loss_value, "sft_lr": lr}, c.device)

    def save(self, path):
        rng = parallel.gather_objects(
            {
                "torch": torch.get_rng_state(),
                "data": self.rng.getstate(),
                "cuda": torch.cuda.get_rng_state(self.cfg.device).cpu()
                if str(self.cfg.device).startswith("cuda")
                else None,
            }
        )
        if parallel.rank():
            return
        payload = {
            "format_version": 1,
            "kind": "offline_sft",
            "config": asdict(self.cfg),
            "actor": self.actor.trainable_state(),
            "optimizer": self.optimizer.state_dict(),
            "sft_step": self.step,
            "phase": "sft",
            "dataset_identity": self.dataset.identity,
            "world_size": parallel.world_size(),
            "rank_rng": rng,
        }
        path = local_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def restore(self, path):
        state = torch.load(local_path(path), map_location="cpu", weights_only=True)
        if state.get("kind") != "offline_sft" or state["world_size"] != parallel.world_size():
            raise ValueError("SFT resume needs an offline SFT checkpoint and the same world size")
        if state["dataset_identity"] != self.dataset.identity:
            raise ValueError("SFT dataset changed")
        ignore = {"output", "device", "sft_steps", "replay", "rounds"}
        from capo.identity import canonical

        for k, v in asdict(self.cfg).items():
            if k not in ignore and canonical(v) != canonical(state["config"].get(k)):
                raise ValueError(f"SFT resume config mismatch: {k}")
        load_actor_state(self.actor, state["actor"], state["config"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state["sft_step"]
        rng = state["rank_rng"][parallel.rank()]
        self.rng.setstate(rng["data"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"], self.cfg.device)


def run(cfg, *, resume=None, save_every=100):
    if not cfg.demonstrations:
        raise ValueError("Set demonstrations to a prepared offline dataset; see capo.demonstrations prepare")
    if cfg.backend == "pi05" and not (cfg.data_spec or {}).get("task"):
        raise ValueError("Use the prepared dataset's data_config.json (bound norm/cameras/task)")
    if cfg.sft_steps < 1:
        raise ValueError("sft_steps must be positive")
    parallel.initialize(cfg)
    if cfg.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(cfg.seed)
    dataset = DemonstrationDataset(cfg)
    actor = make_backbone(cfg)
    trainer = SFTTrainer(cfg, actor, dataset)
    output = local_path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    if parallel.rank() == 0:
        if (output / "sft.pt").exists() and not resume:
            raise FileExistsError("SFT output exists; use --resume or a new output directory")
        cfg.save(output / "config.json")
    torch.manual_seed(cfg.seed + parallel.rank())
    if resume:
        trainer.restore(resume)
    while trainer.step < cfg.sft_steps:
        metrics = {"sft_step": trainer.step + 1, **trainer.update()}
        if parallel.rank() == 0:
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
        if save_every > 0 and trainer.step % save_every == 0:
            trainer.save(output / "latest.pt")
    trainer.save(output / "sft.pt")
    return output / "sft.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    try:
        run(Config.load(args.config), resume=args.resume, save_every=args.save_every)
    finally:
        if parallel.active():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
