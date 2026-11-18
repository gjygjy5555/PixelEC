"""Two fixed-budget paired views for SICE: full-frame rectangle or native pixels."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from .native_data import NativeJsonlDataset, OPTIONS, collate_native, pad_rb
from pixrestore.data import _paired_random_crop


class FixedBudgetViewDataset(NativeJsonlDataset):
    def __init__(
        self,
        manifests: Iterable[str | Path],
        crop_size: int,
        *,
        crop_mode: str = "random",
        max_retries: int = 50,
    ) -> None:
        if crop_mode != "random":
            raise ValueError("Fixed-budget experiment requires random training crops")
        super().__init__(manifests, crop_size, max_retries=max_retries)
        self.view_mode = str(OPTIONS["view_mode"])
        self.global_rect_probability = float(OPTIONS.get("global_rect_probability", 1.0))
        self.area_edge = int(OPTIONS.get("area_edge", 512))
        self.crop_edge = int(OPTIONS.get("crop_edge", 512))
        if self.view_mode not in (
            "global_rect", "native_local", "native_rect_crop", "square_crop",
            "rect_square_mix", "rect_rect_mix",
        ):
            raise ValueError(self.view_mode)
        if not 0.0 <= self.global_rect_probability <= 1.0:
            raise ValueError("global_rect_probability must be in [0, 1]")
        if crop_size != 512 or self.patch_size != 8:
            raise ValueError("Expected the 512-pixel, patch-8 control configuration")

    def _prepare(
        self,
        lq: Image.Image,
        gt: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image, torch.Tensor]:
        if lq.size != gt.size:
            raise ValueError(f"Paired LQ/GT dimensions differ: {lq.size} versus {gt.size}")
        if mask is not None and mask.size != lq.size:
            mask = mask.resize(lq.size, Image.Resampling.NEAREST)

        mode = self.view_mode
        if mode == "rect_square_mix":
            mode = "global_rect" if random.random() < self.global_rect_probability else "square_crop"
        elif mode == "rect_rect_mix":
            mode = "global_rect" if random.random() < self.global_rect_probability else "native_rect_crop"

        if mode == "square_crop":
            lq, gt, mask = _paired_random_crop(lq, gt, self.crop_edge, mask)
        elif mode in ("global_rect", "native_rect_crop"):
            ratio = lq.width / lq.height
            width = max(self.patch_size, round(self.area_edge * math.sqrt(ratio) / self.patch_size) * self.patch_size)
            height = max(self.patch_size, round(self.area_edge / math.sqrt(ratio) / self.patch_size) * self.patch_size)
            if mode == "global_rect":
                size = (width, height)
                lq = lq.resize(size, Image.Resampling.BICUBIC)
                gt = gt.resize(size, Image.Resampling.LANCZOS)
                if mask is not None:
                    mask = mask.resize(size, Image.Resampling.NEAREST)
            else:
                if min(lq.size) < self.patch_size:
                    raise ValueError(f"Image too small for patch-{self.patch_size}: {lq.size}")
                feasible_scale = min(1.0, lq.width / width, lq.height / height)
                width = max(self.patch_size, math.floor(width * feasible_scale / self.patch_size) * self.patch_size)
                height = max(self.patch_size, math.floor(height * feasible_scale / self.patch_size) * self.patch_size)
                left = random.randint(0, lq.width - width)
                top = random.randint(0, lq.height - height)
                box = (left, top, left + width, top + height)
                lq, gt = lq.crop(box), gt.crop(box)
                if mask is not None:
                    mask = mask.crop(box)
        else:
            if min(lq.size) < self.patch_size:
                raise ValueError(f"Image too small for patch-{self.patch_size}: {lq.size}")
            width = min(self.crop_edge, (lq.width // self.patch_size) * self.patch_size)
            height = min(self.crop_edge, (lq.height // self.patch_size) * self.patch_size)
            left = random.randint(0, lq.width - width)
            top = random.randint(0, lq.height - height)
            box = (left, top, left + width, top + height)
            lq, gt = lq.crop(box), gt.crop(box)
            if mask is not None:
                mask = mask.crop(box)

        if self.hflip and random.random() < 0.5:
            lq = lq.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        original_width, original_height = lq.size
        lq, pad_right, pad_bottom = pad_rb(lq, self.patch_size)
        gt, _, _ = pad_rb(gt, self.patch_size)
        invalid = torch.zeros(1, lq.height, lq.width)
        if pad_bottom:
            invalid[:, original_height:, :] = 1.0
        if pad_right:
            invalid[:, :, original_width:] = 1.0
        if mask is not None:
            mask, _, _ = pad_rb(mask.convert("RGB"), self.patch_size)
            annotated = pil_to_tensor(mask)[:1].float().div_(255.0)
            invalid = invalid.maximum(annotated)
        return lq, gt, invalid


__all__ = ["FixedBudgetViewDataset", "OPTIONS", "collate_native"]
