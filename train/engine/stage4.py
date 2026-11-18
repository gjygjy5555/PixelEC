"""Square-view Stage IV training setup."""
from pathlib import Path
import random
import sys

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from safetensors.torch import load_file

from .rgb3_head import RGB3_HEAD_PARAMETERS, attach_rgb3


def run(config, source, active_config):
    import pixrestore.data as paired_data
    from . import train_base as official
    from .common import add_ssim_to_flow, install_square_view_and_virtual_rope
    from .view_data import FixedBudgetViewDataset, OPTIONS, collate_native
    from .laplacian_decoder import set_lq
    from .dino_memory import install_all_layer_dino_memory

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
                raise ValueError("LQ/GT dimensions differ")
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
    parameter_ids: dict[str, set[int]] = {"head": set(), "tail": set()}
    predictions: dict = {}

    def build(cfg):
        model = install_all_layer_dino_memory(original_build(cfg))
        model = attach_rgb3(model)
        for name, parameter in model.named_parameters():
            is_head = name.startswith("final_layer.")
            is_tail = name.startswith("blocks.10.") or name.startswith("blocks.11.")
            parameter.requires_grad_(is_head or is_tail)
            if is_head:
                parameter_ids["head"].add(id(parameter))
            elif is_tail:
                parameter_ids["tail"].add(id(parameter))
        head_count = sum(p.numel() for p in model.parameters() if id(p) in parameter_ids["head"])
        tail_count = sum(p.numel() for p in model.parameters() if id(p) in parameter_ids["tail"])
        if head_count != RGB3_HEAD_PARAMETERS or tail_count <= 0:
            raise RuntimeError(f"Unexpected Stage-IV groups: head={head_count} tail={tail_count}")
        model.register_forward_hook(lambda _module, _inputs, output: predictions.__setitem__("rgb", output[0]))
        print(f"[square-virtual-stage4] route={config['experiment_mode']} head={head_count} "
              f"tail={tail_count} blocks=10,11", flush=True)
        return model

    official.build_model = build

    def strict_load(model, ema, checkpoint):
        path = Path(checkpoint)
        path = path / "ema_model.safetensors" if path.is_dir() else path
        if path != source:
            raise RuntimeError(f"Wrong Stage-III EMA: {path}")
        model.load_state_dict(load_file(str(path)), strict=True)
        ema.load_state_dict(model.state_dict(), strict=True)
        print(f"[square-virtual-stage4] strict Stage-III EMA={source}", flush=True)

    official.load_pretrained = strict_load
    add_ssim_to_flow(official, predictions, original_lq=set_lq)
    original_adam = torch.optim.AdamW

    class GroupedAdam(original_adam):
        def __init__(self, params, *other_args, **kwargs):
            trainable = [parameter for parameter in params if parameter.requires_grad]
            heads = [parameter for parameter in trainable if id(parameter) in parameter_ids["head"]]
            tails = [parameter for parameter in trainable if id(parameter) in parameter_ids["tail"]]
            if len(heads) + len(tails) != len(trainable):
                raise RuntimeError("Trainable parameter escaped the head/tail groups")
            groups = [
                {"params": heads, "lr": float(config["student_learning_rate"])},
                {"params": tails, "lr": float(config["trunk_learning_rate"])},
            ]
            super().__init__(groups, *other_args, **kwargs)

    official.torch.optim.AdamW = GroupedAdam

    sys.argv = [str(official.__file__), "--config", str(active_config)]
    official.main()
