"""PixelEC with all-layer DINO conditioning and a multiscale RGB readout."""
from __future__ import annotations


import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

from vendor.pixrestore import LightningDiT_PixelDiffusion
from vendor.pixrestore.models.layers import modulate
from vendor.pixrestore.models.rope import VisionRotaryEmbeddingFast, broadcat, rotate_half


class VirtualNativeRoPE(VisionRotaryEmbeddingFast):
    """Original 64x64 RoPE buffers, with an optional original-canvas step.

    Only the fixed 64x64 training grid uses this forward.  A native HxW grid
    uses PixRestore's existing dynamic, reference-64 RoPE implementation.
    """

    canvas_hw: tuple[int, int] | None = None

    @torch.compiler.disable
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.canvas_hw is None:
            return super().forward(tokens)
        native_h, native_w = self.canvas_hw
        if min(native_h, native_w) <= 0 or tokens.shape[-2] != 4096:
            raise ValueError("Virtual RoPE expects 64x64 tokens and positive native grid")
        half = tokens.shape[-1] // 2
        freq = 1.0 / 10000 ** (torch.arange(0, half, 2, device=tokens.device).float() / half)
        h = torch.arange(64, device=tokens.device).float() * (64.0 / native_h)
        w = torch.arange(64, device=tokens.device).float() * (64.0 / native_w)
        fh = repeat(torch.einsum("h,f->hf", h, freq), "h n -> h (n r)", r=2)
        fw = repeat(torch.einsum("w,f->wf", w, freq), "w n -> w (n r)", r=2)
        phase = broadcat((fh[:, None, :], fw[None, :, :]), dim=-1)
        cosine = phase.cos().reshape(-1, phase.shape[-1]).to(tokens.dtype)
        sine = phase.sin().reshape(-1, phase.shape[-1]).to(tokens.dtype)
        return (tokens * cosine + rotate_half(tokens) * sine).contiguous()


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.silu(self.conv1(x)))


class _Stage(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.blocks = nn.Sequential(_ResidualBlock(channels), _ResidualBlock(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(F.silu(self.fuse(x)))


class NearestRGB3Renderer(nn.Module):
    """Reconstruct RGB from nearest-upsampled features and four LQ scales."""
    def __init__(self, original, hidden=384, channels=64):
        super().__init__()
        self.norm_final = original.norm_final
        self.adaLN_modulation = original.adaLN_modulation
        self.token_proj = nn.Linear(hidden, channels)
        self.lq8 = nn.Conv2d(3, channels, 3, padding=1)
        self.stage8 = _Stage(channels)
        self.lq4 = nn.Conv2d(6, channels, 3, padding=1)
        self.stage4 = _Stage(channels)
        self.lq2 = nn.Conv2d(6, channels, 3, padding=1)
        self.stage2 = _Stage(channels)
        self.lq1 = nn.Conv2d(6, channels, 3, padding=1)
        self.stage1 = _Stage(channels)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.grid_hw: tuple[int, int] | None = None
        self.original_lq: torch.Tensor | None = None

        # Preserve the RGB3 head initialization and RNG order.
        rng = torch.get_rng_state()
        for s in (4, 2, 1):
            old = getattr(self, f'lq{s}')
            new = nn.Conv2d(3, old.out_channels, 3, padding=1)
            with torch.no_grad():
                new.weight.copy_(old.weight[:, :3] + old.weight[:, 3:])
                new.bias.copy_(old.bias)
            setattr(self, f'lq{s}', new)
        torch.set_rng_state(rng)

    def forward(self,tokens,condition):
        shift,scale=self.adaLN_modulation(condition).chunk(2,1)
        tokens=modulate(self.norm_final(tokens),shift,scale)
        h,w=self.grid_hw;lq=self.original_lq
        f=self.token_proj(tokens).transpose(1,2).reshape(tokens.shape[0],-1,h,w)
        rgb={s:lq if s==1 else F.interpolate(lq,size=(h*8//s,w*8//s),mode='nearest') for s in [1,2,4,8]}
        f=self.stage8(torch.cat((f,self.lq8(rgb[8])),1))
        for s in [4,2,1]:
            f=getattr(self,f'stage{s}')(torch.cat((F.interpolate(f,size=rgb[s].shape[-2:],mode='nearest'),getattr(self,f'lq{s}')(rgb[s])),1))
        return lq+self.out(f)


class PixelEC(LightningDiT_PixelDiffusion):
    """PixelEC backbone and RGB readout."""

    def __init__(self, **backbone):
        super().__init__(**backbone)
        if self.feat_rope is not None:
            old = self.feat_rope
            rope = VirtualNativeRoPE(dim=self.hidden_size // self.num_heads // 2, pt_seq_len=64)
            rope.load_state_dict(old.state_dict(), strict=True)
            self.feat_rope = rope
        self.final_layer = NearestRGB3Renderer(self.final_layer, self.hidden_size)
        self.last_rgb: torch.Tensor | None = None

    def _gate_venc_features(self, z, *args, **kwargs):
        if isinstance(z, (tuple, list)):
            if len(z) != self.num_fused_layers:
                raise ValueError("Incorrect number of DINO levels")
            return torch.cat(tuple(z), dim=1)
        return super()._gate_venc_features(z, *args, **kwargs)

    def unpatchify(self, image, h_p=None, w_p=None):
        return image

    def set_observation(self, original_lq: torch.Tensor, canvas_hw: tuple[int, int] | None = None) -> None:
        self.final_layer.original_lq = original_lq
        if self.feat_rope is not None:
            self.feat_rope.canvas_hw = canvas_hw

    def forward(self, x, t, r=None, z=None, z_hq=None):
        self.final_layer.grid_hw = (x.shape[-2] // 8, x.shape[-1] // 8)
        rgb, features = super().forward(x, t, r=r, z=z, z_hq=z_hq)
        self.last_rgb = rgb
        return rgb, features
