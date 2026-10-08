import math

import torch
from torch import nn
import torch.nn.functional as F


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Linear):

    def __init__(self, base, rank, alpha):

        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return base + update * self.scaling

    @torch.no_grad()
    def merged(self):

        layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        layer.weight.copy_(
            (self.weight.float() + self.scaling * (self.lora_B @ self.lora_A)).to(self.weight.dtype)
        )
        if self.bias is not None:
            layer.bias.copy_(self.bias)
        layer.requires_grad_(False)
        return layer


def inject_expert_lora(expert, rank, alpha):
    expert.requires_grad_(False)
    count = 0
    for module in list(expert.modules()):
        for name, child in list(module.named_children()):
            if name in TARGETS and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                setattr(module, name, LoRALinear(child, rank, alpha))
                count += 1
    if not count:
        raise ValueError("No attention/MLP linear layers found in action expert")
    return count


def merge_expert_lora(expert):
    for module in list(expert.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
