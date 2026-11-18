"""Continuous Laplacian-pyramid pixel renderer for PixRestore.

The DiT owns only an H/8 coarse feature map.  Every RGB pixel is rendered by
shared, overlapping convolutions; no token owns a fixed 8x8 RGB tile.  A fixed
Gaussian/Laplacian pyramid of the *original* LQ image supplies the observed
high-frequency evidence which cannot be recovered by interpolating tokens.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pixrestore.models.layers import modulate


def gaussian_kernel(channels: int, device, dtype) -> torch.Tensor:
    vector = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
    kernel = (vector[:, None] * vector[None, :]) / 256.0
    return kernel.to(dtype=dtype).expand(channels, 1, 5, 5)


def blur(image: torch.Tensor) -> torch.Tensor:
    padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
    weight = gaussian_kernel(image.shape[1], image.device, image.dtype)
    return F.conv2d(padded, weight, groups=image.shape[1])


def down(image: torch.Tensor) -> torch.Tensor:
    return blur(image)[:, :, ::2, ::2]


def expand(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    # Fixed interpolation only transports a coarse state.  Learned 3x3 stages
    # subsequently reconstruct RGB from overlapping neighborhoods and LQ bands.
    return blur(F.interpolate(image, size=size, mode="nearest"))

class Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.conv2(F.silu(self.conv1(feature)))


class Stage(nn.Module):
    def __init__(self, in_channels: int, channels: int = 64):
        super().__init__()
        self.fuse = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(Block(channels), Block(channels))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.blocks(F.silu(self.fuse(feature)))

class LaplacianFinalLayer(nn.Module):
    def __init__(self, old: nn.Module, hidden: int = 384, channels: int = 64):
        super().__init__()
        self.norm_final = old.norm_final
        self.adaLN_modulation = old.adaLN_modulation
        self.token_proj = nn.Linear(hidden, channels)
        self.lq8 = nn.Conv2d(3, channels, 3, padding=1)
        self.stage8 = Stage(channels * 2, channels)
        self.lq4 = nn.Conv2d(6, channels, 3, padding=1)
        self.stage4 = Stage(channels * 2, channels)
        self.lq2 = nn.Conv2d(6, channels, 3, padding=1)
        self.stage2 = Stage(channels * 2, channels)
        self.lq1 = nn.Conv2d(6, channels, 3, padding=1)
        self.stage1 = Stage(channels * 2, channels)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.grid_hw: tuple[int, int] | None = None
        self.original_lq: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if self.grid_hw is None:
            raise RuntimeError("token grid was not captured")
        if self.original_lq is None:
            raise RuntimeError("original LQ was not supplied to the renderer")

        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        tokens = modulate(self.norm_final(tokens), shift, scale)
        h8, w8 = self.grid_hw
        batch = tokens.shape[0]
        feature = self.token_proj(tokens).transpose(1, 2).reshape(batch, -1, h8, w8)
        lq = self.original_lq

        gaussian2 = down(lq)
        gaussian4 = down(gaussian2)
        gaussian8 = down(gaussian4)
        laplacian4 = gaussian4 - expand(gaussian8, gaussian4.shape[-2:])
        laplacian2 = gaussian2 - expand(gaussian4, gaussian2.shape[-2:])
        laplacian1 = lq - expand(gaussian2, lq.shape[-2:])

        feature = self.stage8(torch.cat((feature, self.lq8(gaussian8)), dim=1))
        feature = self.stage4(
            torch.cat(
                (expand(feature, gaussian4.shape[-2:]), self.lq4(torch.cat((gaussian4, laplacian4), dim=1))),
                dim=1,
            )
        )
        feature = self.stage2(
            torch.cat(
                (expand(feature, gaussian2.shape[-2:]), self.lq2(torch.cat((gaussian2, laplacian2), dim=1))),
                dim=1,
            )
        )
        feature = self.stage1(
            torch.cat(
                (expand(feature, lq.shape[-2:]), self.lq1(torch.cat((lq, laplacian1), dim=1))),
                dim=1,
            )
        )
        return lq + self.out(feature)

def install(model: nn.Module) -> nn.Module:
    new = LaplacianFinalLayer(model.final_layer, model.hidden_size)
    model.final_layer = new
    # The new final layer already returns [B,3,H,W].
    model.unpatchify = lambda image, h_p=None, w_p=None: image

    def capture_grid(_module, args):
        height, width = args[0].shape[-2:]
        new.grid_hw = (height // 8, width // 8)

    model.register_forward_pre_hook(capture_grid)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def set_lq(model: nn.Module, lq: torch.Tensor) -> None:
    unwrap(model).final_layer.original_lq = lq
