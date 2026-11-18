"""Attach the RGB3 renderer with the delivery training initialization."""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from .laplacian_decoder import install
from pixrestore.models.layers import modulate

RGB3_HEAD_PARAMETERS = 1_215_619


def forward(self, tokens, condition):
    shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
    tokens = modulate(self.norm_final(tokens), shift, scale)
    height, width = self.grid_hw
    lq = self.original_lq
    features = self.token_proj(tokens).transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
    levels = {
        s: lq if s == 1 else F.interpolate(lq, size=(height * 8 // s, width * 8 // s), mode="nearest")
        for s in (1, 2, 4, 8)
    }
    features = self.stage8(torch.cat((features, self.lq8(levels[8])), dim=1))
    for scale in (4, 2, 1):
        rgb = getattr(self, f"lq{scale}")(levels[scale])
        upsampled = F.interpolate(features, size=levels[scale].shape[-2:], mode="nearest")
        features = getattr(self, f"stage{scale}")(torch.cat((upsampled, rgb), dim=1))
    return lq + self.out(features)


def attach_rgb3(model):
    model = install(model)
    head = model.final_layer
    rng = torch.get_rng_state()
    for scale in (4, 2, 1):
        previous = getattr(head, f"lq{scale}")
        rgb = nn.Conv2d(3, previous.out_channels, 3, padding=1)
        with torch.no_grad():
            rgb.weight.copy_(previous.weight[:, :3] + previous.weight[:, 3:])
            rgb.bias.copy_(previous.bias)
        setattr(head, f"lq{scale}", rgb)
    torch.set_rng_state(rng)
    head.forward = MethodType(forward, head)
    if sum(parameter.numel() for parameter in head.parameters()) != RGB3_HEAD_PARAMETERS:
        raise RuntimeError("Unexpected RGB3 head size")
    return model
