"""Full-resolution paired dataset: no crop, reflect-pad to a patch multiple.

Mirrors `infer_native.py` exactly so training and inference see the same
geometry: original H x W, right/bottom reflect pad to a multiple of patch_size,
never a resize or a content crop. The padded strip is reported through
`invalid_mask`, which `pixrestore.flow` already excludes from the loss.

`_resize_short_edge` is still applied for the rare pair whose short edge is
below the patch multiple; SICE never hits it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from pixrestore.data import _load_mask, _read_manifest, _to_tensor

OPTIONS: dict[str, Any] = {"max_long": 0, "hflip": False}


def cap_long_side(image: Image.Image, cap: int) -> tuple[int, int, int, int] | None:
    """Center-crop box that bounds the long side, or None when already under cap."""
    w, h = image.size
    if cap <= 0 or max(w, h) <= cap:
        return None
    if w >= h:
        left = (w - cap) // 2
        return (left, 0, left + cap, h)
    top = (h - cap) // 2
    return (0, top, w, top + cap)


def pad_rb(image: Image.Image, multiple: int) -> tuple[Image.Image, int, int]:
    """Right/bottom reflect pad to a multiple, identical to infer_native.pad_rb."""
    w, h = image.size
    nw = (w + multiple - 1) // multiple * multiple
    nh = (h + multiple - 1) // multiple * multiple
    pad_r, pad_b = nw - w, nh - h
    if pad_r == 0 and pad_b == 0:
        return image, 0, 0
    arr = np.asarray(image)
    arr = np.pad(arr, ((0, pad_b), (0, pad_r), (0, 0)), mode="reflect")
    return Image.fromarray(arr), pad_r, pad_b


class NativeJsonlDataset(Dataset):
    """Drop-in replacement for PairedJsonlDataset that keeps native resolution.

    The positional signature matches the original, `(manifests, crop_size)`, so
    train.py needs no change; `crop_size` is ignored and module-level OPTIONS
    carries the native-only knobs.
    """

    def __init__(
        self,
        manifests: Iterable[str | Path],
        crop_size: int,
        *,
        max_retries: int = 50,
    ) -> None:
        del crop_size
        self.groups = [_read_manifest(path) for path in manifests]
        if not self.groups:
            raise ValueError("At least one training manifest is required")
        self.max_retries = int(max_retries)
        self.patch_size = int(OPTIONS.get("patch_size", 8))
        self.max_long = int(OPTIONS.get("max_long", 0) or 0)
        self.hflip = bool(OPTIONS.get("hflip", False))
        self.has_masks = True  # padding always produces one
        self.group_length = max(map(len, self.groups))

    def __len__(self) -> int:
        return self.group_length * len(self.groups)

    def _prepare(self, lq: Image.Image, gt: Image.Image, mask: Image.Image | None):
        if gt.size != lq.size:
            gt = gt.resize(lq.size, Image.Resampling.LANCZOS)
        if mask is not None and mask.size != lq.size:
            mask = mask.resize(lq.size, Image.Resampling.NEAREST)

        box = cap_long_side(lq, self.max_long)
        if box is not None:
            lq, gt = lq.crop(box), gt.crop(box)
            mask = mask.crop(box) if mask is not None else None

        if self.hflip and random.random() < 0.5:
            lq = lq.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        orig_w, orig_h = lq.size
        lq, pad_r, pad_b = pad_rb(lq, self.patch_size)
        gt, _, _ = pad_rb(gt, self.patch_size)

        invalid = torch.zeros(1, lq.height, lq.width)
        if pad_b:
            invalid[:, orig_h:, :] = 1.0
        if pad_r:
            invalid[:, :, orig_w:] = 1.0
        if mask is not None:
            mask, _, _ = pad_rb(mask.convert("RGB"), self.patch_size)
            annotated = pil_to_tensor(mask)[:1].float().div_(255.0)
            invalid = invalid.clamp_max(1.0).maximum(annotated)
        return lq, gt, invalid

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_index = index % len(self.groups)
        group = self.groups[group_index]
        pair_index = (index // len(self.groups)) % len(group)
        last_error: Exception | None = None

        for retry in range(self.max_retries):
            lq_path, gt_path, mask_path = group[(pair_index + retry) % len(group)]
            try:
                with Image.open(lq_path) as image:
                    lq = image.convert("RGB")
                with Image.open(gt_path) as image:
                    gt = image.convert("RGB")
                lq, gt, invalid = self._prepare(lq, gt, _load_mask(mask_path))
                return {
                    "lq": _to_tensor(lq),
                    "hq": _to_tensor(gt),
                    "invalid_mask": invalid,
                    "lq_path": lq_path,
                    "gt_path": gt_path,
                }
            except (OSError, ValueError, RuntimeError) as error:
                last_error = error
        raise RuntimeError(f"Failed to load pair after {self.max_retries} attempts") from last_error


def collate_native(batch: list[dict[str, Any]]) -> dict[str, Any]:
    shapes = {tuple(sample["lq"].shape) for sample in batch}
    if len(shapes) > 1:
        raise ValueError(
            "Native batches hold one resolution each; run global_batch_size=1 with "
            f"gradient accumulation. Got {sorted(shapes)}"
        )
    return {
        "lq": torch.stack([sample["lq"] for sample in batch]),
        "hq": torch.stack([sample["hq"] for sample in batch]),
        "invalid_mask": torch.stack([sample["invalid_mask"] for sample in batch]),
        "lq_path": [sample["lq_path"] for sample in batch],
        "gt_path": [sample["gt_path"] for sample in batch],
    }
