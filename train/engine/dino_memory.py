"""All-layer DINO memory used by the training backbone."""
from types import MethodType

import torch
from torch import nn


def install_all_layer_dino_memory(model: nn.Module) -> nn.Module:
    """Bypass global early fusion; concatenate six DINO layers as CA memory."""

    def all_layer_gate(
        self,
        z,
        x,
        c=None,
        block_idx=0,
        deg_token=None,
        z_teacher=None,
        z_hq_teacher=None,
        shared_prior_probs=None,
        compute_prior_loss=True,
    ):
        del x, c, block_idx, deg_token, z_teacher, z_hq_teacher
        del shared_prior_probs, compute_prior_loss
        if isinstance(z, (list, tuple)):
            if len(z) != self.num_fused_layers:
                raise ValueError(f"expected {self.num_fused_layers} DINO layers, got {len(z)}")
            return torch.cat(tuple(z), dim=1)
        return z

    model._gate_venc_features = MethodType(all_layer_gate, model)
    model.experiment_mode = "all_layer_dino_memory"
    return model
