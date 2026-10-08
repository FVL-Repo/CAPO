from dataclasses import dataclass
import math
from collections.abc import Callable

import torch
from torch import Tensor

Velocity = Callable[[Tensor, Tensor], Tensor]


def cutoff_index(cutoff: float, steps: int, rule: str = "ceil") -> int:
    if steps < 1 or not 0 <= cutoff < 1:
        raise ValueError("Require steps >= 1 and 0 <= cutoff < 1")
    position = cutoff * steps
    if rule == "error":
        if abs(position - round(position)) > 1e-6:
            raise ValueError("Cutoff is off-grid; choose explicit ceil/floor rounding")
        index = round(position)
    elif rule == "ceil":
        index = math.ceil(position - 1e-8)
    elif rule == "floor":
        index = math.floor(position + 1e-8)
    else:
        raise ValueError(f"Unknown cutoff rule {rule}")
    if index >= steps:
        raise ValueError("Rounded terminal interval is empty")
    return index


@torch.no_grad()
def rollout(velocity: Velocity, noise: Tensor, steps: int) -> Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    states = [noise.detach()]
    for i in range(steps):
        tau = noise.new_full((len(noise),), i / steps)
        states.append(states[-1] + velocity(states[-1], tau) / steps)
    return torch.stack(states).detach()


@dataclass
class Targets:
    states: Tensor
    velocities: dict[int, Tensor]
    costates: dict[int, Tensor]
    first_index: int


def build_targets(
    reference: Velocity,
    states: Tensor,
    value: Callable[[Tensor], Tensor],
    *,
    eta=1.0,
    alpha_q=1.0,
    cutoff=0.5,
    cutoff_rule="ceil",
) -> Targets:
    states = states.detach()
    steps = len(states) - 1
    first = cutoff_index(cutoff, steps, cutoff_rule)
    with torch.enable_grad():
        terminal = states[-1].detach().requires_grad_(True)
        q = value(terminal)
        c = -alpha_q * torch.autograd.grad(q.sum(), terminal)[0]
    c = c.detach()
    targets, costates = {}, {steps: c}
    for i in range(steps - 1, first - 1, -1):
        with torch.enable_grad():
            x = states[i].detach().requires_grad_(True)
            tau = x.new_full((len(x),), i / steps)
            v = reference(x, tau)
            targets[i] = (v - eta * c).detach()
            if v.requires_grad:
                vjp = torch.autograd.grad(v, x, c, allow_unused=True)[0]
            else:
                vjp = None
        c = (c if vjp is None else c + vjp / steps).detach()
        costates[i] = c
    return Targets(states, targets, costates, first)


def squared_norm_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    residual = (prediction.float() - target.float()).square()
    if mask is not None:
        residual = residual * mask
    return residual.flatten(1).sum(1).mean()


def flow_matching_loss(
    velocity: Velocity, actions: Tensor, mask: Tensor, noise: Tensor | None = None, tau: Tensor | None = None
) -> Tensor:
    noise = torch.randn_like(actions) if noise is None else noise
    tau = torch.rand(len(actions), device=actions.device) if tau is None else tau
    x = (1 - tau[:, None, None]) * noise + tau[:, None, None] * actions
    return squared_norm_loss(velocity(x, tau), actions - noise, mask)
