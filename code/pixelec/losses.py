"""The additional Stage-2/3 structural loss; FM/DINO remain PixRestore's."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def ssim_rgb(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Exact 11x11, sigma=1.5, RGB grouped-convolution implementation used in
    # the locked virtual-RoPE runs. Inputs are model-space [-1,1].
    prediction = (prediction.float() + 1) * 0.5
    target = (target.float() + 1) * 0.5
    coords = torch.arange(11, device=prediction.device, dtype=torch.float32) - 5
    gaussian = torch.exp(-coords.square() / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)[None, None].expand(3, 1, 11, 11).contiguous()
    mu_p = F.conv2d(prediction, window, groups=3)
    mu_t = F.conv2d(target, window, groups=3)
    sigma_p = F.conv2d(prediction.square(), window, groups=3) - mu_p.square()
    sigma_t = F.conv2d(target.square(), window, groups=3) - mu_t.square()
    sigma_pt = F.conv2d(prediction * target, window, groups=3) - mu_p * mu_t
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p.square() + mu_t.square() + c1) * (sigma_p + sigma_t + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean()
