"""Square-view Stage II and III training setup."""
from pathlib import Path
import random
import sys

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from safetensors.torch import load_file

from .rgb3_head import RGB3_HEAD_PARAMETERS, attach_rgb3


def run(config, source, active_config, stage):
    import pixrestore.data as paired_data
    from . import train_base as official
    from .view_data import FixedBudgetViewDataset, OPTIONS, collate_native
    from .dino_memory import install_all_layer_dino_memory
    from .laplacian_decoder import set_lq
    from .common import install_square_view_and_virtual_rope, add_ssim_to_flow

    install_square_view_and_virtual_rope(official, config)
    OPTIONS.update(patch_size=8, max_long=0, hflip=bool(config["native_hflip"]),
                   view_mode="global_rect", global_rect_probability=1.0,
                   area_edge=512, crop_edge=512)

    class FullSquareDataset(FixedBudgetViewDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            with Image.open(sample["lq_path"]) as original:
                width, height = original.size
            sample["native_token_hw"] = torch.tensor(
                [(height + 7) // 8, (width + 7) // 8], dtype=torch.int32
            )
            return sample

        def _prepare(self, lq: Image.Image, gt: Image.Image, mask: Image.Image | None):
            if lq.size != gt.size:
                raise ValueError(f"Unaligned LQ/GT: {lq.size} != {gt.size}")
            if mask is not None and mask.size != lq.size:
                mask = mask.resize(lq.size, Image.Resampling.NEAREST)
            lq = lq.resize((512, 512), Image.Resampling.BICUBIC)
            gt = gt.resize((512, 512), Image.Resampling.LANCZOS)
            if mask is not None:
                mask = mask.resize((512, 512), Image.Resampling.NEAREST)
            if self.hflip and random.random() < 0.5:
                lq = lq.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if mask is not None:
                    mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            invalid = torch.zeros(1, 512, 512)
            if mask is not None:
                invalid = invalid.maximum(pil_to_tensor(mask.convert("RGB"))[:1].float() / 255.0)
            return lq, gt, invalid

    def collate_with_canvas(batch):
        result = collate_native(batch)
        result["native_token_hw"] = torch.stack([sample["native_token_hw"] for sample in batch])
        return result

    paired_data.PairedJsonlDataset = FullSquareDataset
    paired_data.collate_pairs = collate_with_canvas
    official.PairedJsonlDataset = FullSquareDataset
    official.collate_pairs = collate_with_canvas
    original_build = official.build_model
    predictions = {}

    def build(cfg):
        model = install_all_layer_dino_memory(original_build(cfg))
        if stage == 3:
            model = attach_rgb3(model)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("final_layer."))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable <= 0 or (stage == 3 and trainable != RGB3_HEAD_PARAMETERS):
            raise RuntimeError(f"Unexpected renderer parameter count: {trainable}")
        model.register_forward_hook(lambda _module, _inputs, output: predictions.__setitem__("rgb", output[0]))
        print(f"[square-virtual-stage{stage}] route={config['experiment_mode']} frozen_DiT={stage == 3} "
              f"trainable_parameters={trainable}", flush=True)
        return model

    official.build_model = build

    def strict_load(model, ema, checkpoint):
        path = Path(checkpoint)
        path = path / "ema_model.safetensors" if path.is_dir() else path
        if path != source:
            raise RuntimeError(f"Wrong Stage-II EMA: {path}")
        state = load_file(str(path))
        if "local_gate_logit_scale" in state and "local_gate_logit_scale" not in model.state_dict():
            state.pop("local_gate_logit_scale")
        if stage == 2:
            model.load_state_dict(state, strict=True)
            ema.load_state_dict(model.state_dict(), strict=True)
            return
        old_head = {key for key in state if key.startswith("final_layer.linear.")}
        if old_head != {"final_layer.linear.weight", "final_layer.linear.bias"}:
            raise RuntimeError(f"Unexpected Linear RGB keys: {old_head}")
        for key in old_head:
            state.pop(key)
        missing, unexpected = model.load_state_dict(state, strict=False)
        expected_missing = {
            key for key in model.state_dict()
            if key.startswith("final_layer.")
            and not key.startswith("final_layer.norm_final.")
            and not key.startswith("final_layer.adaLN_modulation.")
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(f"Stage-II/III load mismatch: {missing=} {unexpected=}")
        ema.load_state_dict(model.state_dict(), strict=True)
        print(f"[square-virtual-stage{stage}] strict Stage-II EMA={source}", flush=True)

    official.load_pretrained = strict_load
    add_ssim_to_flow(official, predictions, original_lq=set_lq)

    sys.argv = [str(official.__file__), "--config", str(active_config)]
    official.main()
