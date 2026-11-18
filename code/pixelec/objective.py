"""Inherited FM/DINO objective and the paired RGB structural term."""
import torch
from vendor.pixrestore import PixelDiffusion
from .losses import ssim_rgb

def build_flow(cfg, accelerator):
    return PixelDiffusion(
        flow_ratio=cfg.flow_ratio, time_dist=cfg.time_distribution,
        alpha=cfg.alpha, z_start="noise", cfg_ratio=cfg.cfg_ratio,
        cfg_scale=cfg.cfg_scale, image_size=512, channels=3,
        interp_type="lin", uncond_type="zero", norm_p=cfg.adaptive_loss_power,
        accelerator=accelerator, t_start=0, t_end=1, use_cos=False, args=cfg,
    )

def supervised_loss(flow, model, lq, gt, lq_features, gt_features, feature_fn, raw, invalid_mask=None):
    total, backward, _ = flow.loss_fm(
        model, lq, gt, z=lq_features, z_hq=gt_features, dino_loss_fn=feature_fn, invalid_mask=invalid_mask,
    )
    if flow.args.ssim_loss_weight:
        with torch.autocast(device_type=lq.device.type, enabled=False):
            structural = 1 - ssim_rgb(raw.last_rgb, gt)
        total = total + flow.args.ssim_loss_weight * structural
        backward = total / ((total + flow.norm_eps) ** flow.norm_p).detach()
    raw.last_rgb = None
    return total, backward
