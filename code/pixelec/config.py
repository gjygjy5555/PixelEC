"""Strict-compatible PixelEC model construction."""
from .model import PixelEC


def build_model(cfg, *, state=None):
    if cfg.fusion != "all_layers" or cfg.head != "nearest_rgb3":
        raise ValueError("Expected the main PixelEC model architecture")
    # Retain parameter slots required by the historical checkpoints.
    degradation = bool(getattr(cfg, "use_lq_degradation_token", True))
    local_scale = bool(getattr(cfg, "register_local_gate_logit_scale", False))
    if state is not None:
        degradation = any(k.startswith("lq_degradation_encoder.") for k in state)
        local_scale = "local_gate_logit_scale" in state
    return PixelEC(
        input_size=512, patch_size=8,
        in_channels=6, out_channels=3, hidden_size=384, depth=12,
        num_heads=6, mlp_ratio=4, use_qknorm=True, use_swiglu=True,
        use_rope=True, use_rmsnorm=True, adain_single=True, z_dims=384,
        encdim_ratio=3, num_fused_layers=6, pca_dim=768,
        use_bottleneck_patch_embed=True, use_dino_layer_router=True,
        dino_layer_router_mode="content", same_condition_layer_weights=False,
        gate_temperature=1.0, feature_norm="channel_rmsnorm",
        use_lq_degradation_token=degradation,
        register_local_gate_logit_scale=local_scale,
    )
