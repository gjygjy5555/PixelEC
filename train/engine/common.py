"""Virtual RoPE, paired training views and RGB SSIM loss."""
from __future__ import annotations

import torch
from einops import repeat
from PIL import Image


def install_square_view_and_virtual_rope(official, config: dict) -> None:
    """Exactly one 512 crop per sample; original canvas affects RoPE only."""
    import pixrestore.data as paired_data
    from pixrestore.models.rope import VisionRotaryEmbeddingFast, broadcat, rotate_half
    from .view_data import FixedBudgetViewDataset, OPTIONS, collate_native

    OPTIONS.update(patch_size=8, max_long=0, hflip=True,
                   view_mode="square_crop", global_rect_probability=0.5,
                   area_edge=512, crop_edge=512)

    class SquareCanvasDataset(FixedBudgetViewDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            with Image.open(sample["lq_path"]) as source:
                width, height = source.size
            sample["native_token_hw"] = torch.tensor(
                [(height + 7) // 8, (width + 7) // 8], dtype=torch.int32
            )
            return sample

    def collate(batch):
        result = collate_native(batch)
        result["native_token_hw"] = torch.stack([s["native_token_hw"] for s in batch])
        return result

    paired_data.PairedJsonlDataset = SquareCanvasDataset
    paired_data.collate_pairs = collate
    official.PairedJsonlDataset = SquareCanvasDataset
    official.collate_pairs = collate

    original_forward = VisionRotaryEmbeddingFast.forward

    @torch.compiler.disable
    def virtual_forward(self, tokens):
        canvas = getattr(self, "virtual_native_hw", None)
        if canvas is None:
            return original_forward(self, tokens)
        native_h, native_w = canvas
        if tokens.shape[-2] != 64 * 64:
            raise RuntimeError("Expected fixed 64x64 image-token training grid")
        half = tokens.shape[-1] // 2
        freqs = 1.0 / 10000 ** (
            torch.arange(0, half, 2, device=tokens.device).float() / half
        )
        h = torch.arange(64, device=tokens.device).float() * (64.0 / native_h)
        w = torch.arange(64, device=tokens.device).float() * (64.0 / native_w)
        fh = repeat(torch.einsum("h,f->hf", h, freqs), "h n -> h (n r)", r=2)
        fw = repeat(torch.einsum("w,f->wf", w, freqs), "w n -> w (n r)", r=2)
        phase = broadcat((fh[:, None, :], fw[None, :, :]), dim=-1)
        cosine = phase.cos().reshape(-1, phase.shape[-1]).to(tokens.dtype)
        sine = phase.sin().reshape(-1, phase.shape[-1]).to(tokens.dtype)
        return (tokens * cosine + rotate_half(tokens) * sine).contiguous()

    VisionRotaryEmbeddingFast.forward = virtual_forward
    print("[msec-auto3] 512 square crop, original-canvas virtual RoPE stride", flush=True)


def ssim_rgb(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    prediction = (prediction.float() + 1.0) * 0.5
    target = (target.float() + 1.0) * 0.5
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


def add_ssim_to_flow(official, predictions: dict, *, original_lq=None) -> None:
    original_build = official.build_flow

    def build_flow(cfg, accelerator):
        flow = original_build(cfg, accelerator)
        base_loss = flow.loss_fm
        base_sample = flow.sample_multistep_fm

        def loss(model, lq, hq, *args, **kwargs):
            if original_lq is not None:
                original_lq(model, lq)
            predictions.pop("rgb", None)
            reported, _backward, auxiliary = base_loss(model, lq, hq, *args, **kwargs)
            rgb = predictions.pop("rgb", None)
            if rgb is None:
                raise RuntimeError("Model forward did not expose RGB prediction")
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                structure = 1.0 - ssim_rgb(rgb, hq)
            total = reported + 0.1 * structure
            backward = total / ((total + flow.norm_eps) ** flow.norm_p).detach()
            if not getattr(flow, "_msec_auto3_logged", False):
                print(f"[msec-auto3] FM+0.5 equal-DINO+0.1 SSIM; "
                      f"first_ssim_term={structure.item():.5f}; teacher_weight=0", flush=True)
                flow._msec_auto3_logged = True
            return total, backward, auxiliary

        def sample(model, lq, *args, **kwargs):
            if original_lq is not None:
                original_lq(model, lq)
            return base_sample(model, lq, *args, **kwargs)

        flow.loss_fm = loss
        flow.sample_multistep_fm = sample
        return flow

    official.build_flow = build_flow
