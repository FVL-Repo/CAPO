import torch

from capo.adjoint import rollout
from capo.backbone import make_backbone
from capo.config import local_path
from capo.batching import tree_map
from capo.checkpoints import load_actor_state


class Policy:
    """Unchanged full Euler sampler. No critic is constructed at deployment."""

    def __init__(self, cfg, checkpoint=None, processor=None):
        self.cfg, self.processor = cfg, processor
        self.actor = make_backbone(cfg)
        if checkpoint:
            payload = torch.load(local_path(checkpoint), map_location="cpu", weights_only=True)

            for key in (
                "backend",
                "checkpoint",
                "action_horizon",
                "execution_horizon",
                "action_dim",
                "robot_action_dim",
            ):
                if payload["config"][key] != getattr(cfg, key):
                    raise ValueError(f"Inference config differs from checkpoint: {key}")
            load_actor_state(self.actor, payload["actor"], payload["config"])
        self.actor.eval()

    @torch.no_grad()
    def normalized_actions(self, observation, noise=None):
        obs = tree_map(lambda x: torch.as_tensor(x).unsqueeze(0).to(self.cfg.device), observation)
        ctx = self.actor.prepare(obs)
        if noise is None:
            noise = torch.randn(1, self.cfg.action_horizon, self.cfg.action_dim, device=self.cfg.device)
        return rollout(lambda x, t: self.actor.velocity(ctx, x, t), noise, self.cfg.flow_steps)[-1][0].cpu()

    def infer(self, observation):
        obs = self.processor.observation(observation) if self.processor else observation
        actions = self.normalized_actions(obs).numpy()
        if self.processor:
            actions = self.processor.unnormalize_actions(actions)
        return {"actions": actions}

    def reset(self):
        pass
