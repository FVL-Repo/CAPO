import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

from capo.config import local_path

SOURCE_SHA256 = "175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b"
WEIGHTS_SHA256 = "1d6347f9334a05e051e7bf9bbefb5c3c5cbfd14cc045792deb344a4fadd91d5e"


def same_pad(x, kernel, stride, value=0.0):
    height, width = x.shape[-2:]
    ph = max((math.ceil(height / stride) - 1) * stride + kernel - height, 0)
    pw = max((math.ceil(width / stride) - 1) * stride + kernel - width, 0)
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=value)


class SameConv(nn.Conv2d):
    def forward(self, x):
        return super().forward(same_pad(x, self.kernel_size[0], self.stride[0]))


class ResNetBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv1 = SameConv(inputs, outputs, 3, stride=stride, bias=False)
        self.norm1 = nn.GroupNorm(4, outputs, eps=1e-5)
        self.conv2 = SameConv(outputs, outputs, 3, bias=False)
        self.norm2 = nn.GroupNorm(4, outputs, eps=1e-5)
        self.projection = (
            SameConv(inputs, outputs, 1, stride=stride, bias=False) if inputs != outputs else None
        )
        self.projection_norm = nn.GroupNorm(4, outputs, eps=1e-5) if inputs != outputs else None

    def forward(self, x):
        residual = x if self.projection is None else self.projection_norm(self.projection(x))
        y = F.relu(self.norm1(self.conv1(x)))
        return F.relu(residual + self.norm2(self.conv2(y)))


class ResNet10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_init = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.norm_init = nn.GroupNorm(4, 64, eps=1e-5)
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(64, 64, 1),
                ResNetBlock(64, 128, 2),
                ResNetBlock(128, 256, 2),
                ResNetBlock(256, 512, 2),
            ]
        )

    def forward(self, normalized_images):
        x = F.relu(self.norm_init(self.conv_init(normalized_images)))
        x = F.max_pool2d(same_pad(x, 3, 2, value=float("-inf")), 3, stride=2)
        for block in self.blocks:
            x = block(x)
        return x

    def load_pretrained(self, path):
        path = local_path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing pretrained critic encoder: {path}. See scripts/convert_resnet10.py"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise ValueError(f"ResNet-10 checkpoint checksum mismatch: {path}")
        self.load_state_dict(load_file(str(path)), strict=True)
        return self.requires_grad_(False).eval()


class SpatialLearnedEmbeddings(nn.Module):
    """Per-camera [H,W,C,F] weights; sum spatially, flatten in C,F order."""

    def __init__(self, features=8, bottleneck=256):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(7, 7, 512, features))
        std = (7 * 7 * 512) ** -0.5 / 0.87962566
        nn.init.trunc_normal_(self.kernel, std=std, a=-2 * std, b=2 * std)
        self.bottleneck = nn.Sequential(
            nn.Linear(512 * features, bottleneck), nn.LayerNorm(bottleneck, eps=1e-6), nn.Tanh()
        )

        dense_std = (512 * features) ** -0.5 / 0.87962566
        nn.init.trunc_normal_(self.bottleneck[0].weight, std=dense_std, a=-2 * dense_std, b=2 * dense_std)
        nn.init.zeros_(self.bottleneck[0].bias)

    def forward(self, x):
        pooled = torch.einsum("bchw,hwcf->bcf", x, self.kernel).flatten(1)
        return self.bottleneck(pooled)
