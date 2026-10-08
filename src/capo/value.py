import torch
from torch import nn
import torch.nn.functional as F

from capo.resnet import ResNet10, SpatialLearnedEmbeddings


class Critic(nn.Module):
    architecture = "capo_direct_q_224_v2"

    def __init__(self, cfg):
        super().__init__()
        self.image_keys = tuple(cfg.image_keys)
        self.execution_horizon = cfg.execution_horizon
        self.action_dim = cfg.robot_action_dim
        self.inactive_action_dims = tuple(cfg.inactive_action_dims)
        self.encoder = ResNet10().load_pretrained(cfg.critic_resnet_weights) if self.image_keys else None
        self.camera_encoders = nn.ModuleDict(
            {
                key: SpatialLearnedEmbeddings(cfg.critic_spatial_blocks, cfg.critic_visual_dim)
                for key in self.image_keys
            }
        )
        self.proprio = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.critic_proprio_dim),
            nn.LayerNorm(cfg.critic_proprio_dim, eps=1e-6),
            nn.Tanh(),
        )
        nn.init.xavier_uniform_(self.proprio[0].weight)
        nn.init.zeros_(self.proprio[0].bias)
        feature_dim = cfg.critic_proprio_dim + len(self.image_keys) * cfg.critic_visual_dim
        action_size = cfg.execution_horizon * cfg.robot_action_dim

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim + 2 * action_size, cfg.critic_hidden),
                    nn.LayerNorm(cfg.critic_hidden, eps=1e-6),
                    nn.Tanh(),
                    nn.Linear(cfg.critic_hidden, cfg.critic_hidden),
                    nn.LayerNorm(cfg.critic_hidden, eps=1e-6),
                    nn.Tanh(),
                    nn.Linear(cfg.critic_hidden, 1),
                )
                for _ in range(cfg.critic_heads)
            ]
        )
        for head in self.heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        if self.encoder is not None:
            self.encoder.requires_grad_(False)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    @staticmethod
    def preprocess_image(image):
        if image.ndim != 4:
            raise ValueError("Critic images require a batch dimension")
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != 3:
            raise ValueError("Expected RGB CHW or HWC images")

        image = image.float() / 255 if image.dtype == torch.uint8 else (image.float() + 1) / 2
        image = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False, antialias=True)
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (image - mean) / std

    def encode(self, obs):
        features = []
        for key in self.image_keys:
            valid = obs["image_masks"][key].bool()
            with torch.no_grad():
                image = self.preprocess_image(obs["images"][key])
                image = torch.where(valid[:, None, None, None], image, 0)
                visual = self.encoder(image).detach()
            feature = self.camera_encoders[key](visual)
            features.append(torch.where(valid[:, None], feature, 0))
        features.append(self.proprio(obs["state"].float()))
        return torch.cat(features, -1)

    def from_features(self, features, actions, mask=None):
        if actions.ndim != 3 or actions.shape[1:] != (self.execution_horizon, self.action_dim):
            raise ValueError("Critic actions must be [batch, execution_horizon, robot_action_dim]")
        if mask is not None and mask.shape != actions.shape:
            raise ValueError("Action mask must have the same shape as actions")
        mask = torch.ones_like(actions, dtype=torch.bool) if mask is None else mask.bool()
        if self.inactive_action_dims:
            mask = mask.clone()
            mask[..., list(self.inactive_action_dims)] = False
        executed = torch.where(mask, actions.float(), 0)
        x = torch.cat([features, executed.flatten(1), mask.to(features).flatten(1)], dim=-1)
        return torch.cat([head(x) for head in self.heads], dim=-1)

    def forward(self, obs, actions, mask=None):
        return self.from_features(self.encode(obs), actions, mask)


def td_target(rewards, terminated, durations, next_values, gamma):
    mean = next_values.mean(dim=tuple(range(1, next_values.ndim)))
    return (rewards + (~terminated.bool()).to(rewards) * gamma**durations * mean).detach()


def ranking_loss(corrected_values, rejected_values, margin=0.1):
    return (margin + rejected_values - corrected_values).clamp_min(0).square().mean()


@torch.no_grad()
def polyak_update(target, online, retention):
    for t, p in zip(target.parameters(), online.parameters(), strict=True):
        if p.requires_grad:
            t.lerp_(p, 1 - retention)

    for t, p in zip(target.buffers(), online.buffers(), strict=True):
        t.copy_(p)
