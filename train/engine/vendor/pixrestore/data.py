"""Paired LQ/GT datasets described by JSON Lines manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def _read_manifest(path: str | Path) -> list[tuple[str, str, str | None]]:
    pairs = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            lq = row.get("lq") or row.get("LQ")
            gt = row.get("gt") or row.get("GT")
            mask = row.get("invalid_mask") or row.get("INVALID_MASK")
            if lq and gt:
                pairs.append((str(lq), str(gt), str(mask) if mask else None))
    if not pairs:
        raise ValueError(f"No valid LQ/GT pairs found in {path}")
    return pairs


def _resize_short_edge(image: Image.Image, edge: int, resample: int) -> Image.Image:
    if min(image.size) >= edge:
        return image
    scale = edge / min(image.size)
    size = tuple(round(value * scale) for value in image.size)
    return image.resize(size, resample)


def _paired_crop(
    lq: Image.Image,
    gt: Image.Image,
    edge: int,
    mask: Image.Image | None,
    mode: str = "random",
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    if gt.size != lq.size:
        gt = gt.resize(lq.size, Image.Resampling.LANCZOS)
    if mask is not None and mask.size != lq.size:
        mask = mask.resize(lq.size, Image.Resampling.NEAREST)

    lq = _resize_short_edge(lq, edge, Image.Resampling.BICUBIC)
    gt = gt.resize(lq.size, Image.Resampling.LANCZOS)
    if mask is not None:
        mask = mask.resize(lq.size, Image.Resampling.NEAREST)

    if mode == "center":
        left = max(0, (lq.width - edge) // 2)
        top = max(0, (lq.height - edge) // 2)
    else:
        left = random.randint(0, lq.width - edge)
        top = random.randint(0, lq.height - edge)
    box = (left, top, left + edge, top + edge)
    return lq.crop(box), gt.crop(box), mask.crop(box) if mask else None


def _paired_random_crop(
    lq: Image.Image,
    gt: Image.Image,
    edge: int,
    mask: Image.Image | None,
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    return _paired_crop(lq, gt, edge, mask, mode="random")


def _load_mask(path: str | None) -> Image.Image | None:
    if path is None:
        return None
    mask = np.squeeze(np.load(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D invalid mask, got {mask.shape}")
    return Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor(image).float().div_(127.5).sub_(1.0)


class PairedJsonlDataset(Dataset):
    """Samples manifests equally, then applies aligned LQ/GT crops.

    Manifest rows use ``{"lq": "...", "gt": "..."}`` and may include an
    ``invalid_mask`` NumPy file.

    crop_mode:
      random — training default (new crop each access)
      center — fixed center crop (for fixed-batch observability replay)
    """

    def __init__(
        self,
        manifests: Iterable[str | Path],
        crop_size: int,
        *,
        max_retries: int = 50,
        crop_mode: str = "random",
    ) -> None:
        self.groups = [_read_manifest(path) for path in manifests]
        if not self.groups:
            raise ValueError("At least one training manifest is required")
        self.crop_size = int(crop_size)
        self.max_retries = int(max_retries)
        self.crop_mode = str(crop_mode)
        if self.crop_mode not in ("random", "center"):
            raise ValueError(f"crop_mode must be 'random' or 'center', got {crop_mode!r}")
        self.has_masks = any(mask for group in self.groups for _, _, mask in group)
        self.group_length = max(map(len, self.groups))

    def __len__(self) -> int:
        return self.group_length * len(self.groups)

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
                mask = _load_mask(mask_path)
                lq, gt, mask = _paired_crop(lq, gt, self.crop_size, mask, mode=self.crop_mode)

                sample = {
                    "lq": _to_tensor(lq),
                    "hq": _to_tensor(gt),
                    "lq_path": lq_path,
                    "gt_path": gt_path,
                }
                if self.has_masks:
                    sample["invalid_mask"] = (
                        pil_to_tensor(mask).float().div_(255.0)
                        if mask is not None
                        else torch.zeros(1, self.crop_size, self.crop_size)
                    )
                return sample
            except (OSError, ValueError, RuntimeError) as error:
                last_error = error
        raise RuntimeError(f"Failed to load pair after {self.max_retries} attempts") from last_error


def collate_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "lq": torch.stack([sample["lq"] for sample in batch]),
        "hq": torch.stack([sample["hq"] for sample in batch]),
        "lq_path": [sample["lq_path"] for sample in batch],
        "gt_path": [sample["gt_path"] for sample in batch],
    }
    if "invalid_mask" in batch[0]:
        result["invalid_mask"] = torch.stack([sample["invalid_mask"] for sample in batch])
    return result


def scene_of(stem: str) -> str:
    """'a0053-kme_103_P1.5' -> 'a0053-kme_103'."""
    for suffix in ("_P1.5", "_N1.5", "_P1", "_N1", "_0"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem.rsplit("_", 1)[0]


def group_by_scene(
    rows: list[tuple[str, str, str | None]],
) -> list[list[tuple[str, str, str | None]]]:
    """Group manifest rows into exposure families sharing a scene id."""
    buckets: dict[str, list[tuple[str, str, str | None]]] = {}
    order: list[str] = []
    for row in rows:
        key = scene_of(Path(row[0]).stem)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    return [buckets[k] for k in order]


class SceneGroupedJsonlDataset(Dataset):
    """One item == one scene's full exposure family, cropped identically.

    The exposure family of a scene is the same photograph under different
    exposures; the spatial control field is known to be strongly shared within
    a family (oracle +13 dB shared across EVs vs +0.4 dB across scenes). Batching
    whole families with an *aligned* crop forces the controller gradient to
    explain the exposure difference using structure that is consistent across
    EVs, instead of memorising one exposure at a time.

    ``__getitem__`` returns stacked tensors of shape [n_ev, C, H, W].
    """

    def __init__(
        self,
        manifests: Iterable[str | Path],
        crop_size: int,
        *,
        max_retries: int = 50,
        crop_mode: str = "random",
        scenes_per_batch: int = 1,
        evs_per_scene: int = 5,
    ) -> None:
        self.groups: list[list[list[tuple[str, str, str | None]]]] = []
        for path in manifests:
            self.groups.append(group_by_scene(_read_manifest(path)))
        # merge all manifests into one flat list of families
        families = [fam for g in self.groups for fam in g]
        if not families:
            raise ValueError("At least one training manifest is required")
        self.families = families
        self.crop_size = int(crop_size)
        self.max_retries = int(max_retries)
        self.crop_mode = str(crop_mode)
        if self.crop_mode not in ("random", "center"):
            raise ValueError(f"crop_mode must be 'random' or 'center', got {crop_mode!r}")
        self.scenes_per_batch = max(1, int(scenes_per_batch))
        self.evs_per_scene = max(1, int(evs_per_scene))
        self.has_masks = any(m for fam in families for _, _, m in fam)

    def __len__(self) -> int:
        return max(1, len(self.families) // self.scenes_per_batch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = (index * self.scenes_per_batch) % len(self.families)
        fams = [self.families[(start + j) % len(self.families)]
                for j in range(self.scenes_per_batch)]

        lqs, hqs, masks, lq_paths, gt_paths = [], [], [], [], []
        for fam in fams:
            # aligned crop box shared by every exposure of the scene
            box = None
            for row in fam[: self.evs_per_scene]:
                lq_path, gt_path, mask_path = row
                for retry in range(self.max_retries):
                    try:
                        with Image.open(lq_path) as image:
                            lq = image.convert("RGB")
                        with Image.open(gt_path) as image:
                            gt = image.convert("RGB")
                        mask = _load_mask(mask_path)
                        if gt.size != lq.size:
                            gt = gt.resize(lq.size, Image.Resampling.LANCZOS)
                        lq_r = _resize_short_edge(lq, self.crop_size, Image.Resampling.BICUBIC)
                        gt_r = gt.resize(lq_r.size, Image.Resampling.LANCZOS)
                        if box is None:
                            if self.crop_mode == "center":
                                left = max(0, (lq_r.width - self.crop_size) // 2)
                                top = max(0, (lq_r.height - self.crop_size) // 2)
                            else:
                                left = random.randint(0, lq_r.width - self.crop_size)
                                top = random.randint(0, lq_r.height - self.crop_size)
                            box = (left, top, left + self.crop_size, top + self.crop_size)
                        lq_c = lq_r.crop(box)
                        gt_c = gt_r.crop(box)
                        lqs.append(_to_tensor(lq_c))
                        hqs.append(_to_tensor(gt_c))
                        lq_paths.append(lq_path)
                        gt_paths.append(gt_path)
                        if self.has_masks:
                            if mask is not None:
                                mask_r = mask.resize(lq_r.size, Image.Resampling.NEAREST).crop(box)
                                masks.append(pil_to_tensor(mask_r).float().div_(255.0))
                            else:
                                masks.append(torch.zeros(1, self.crop_size, self.crop_size))
                        break
                    except (OSError, ValueError, RuntimeError):
                        if retry == self.max_retries - 1:
                            raise
                        box = None  # resample the crop on retry
        sample = {
            "lq": torch.stack(lqs),
            "hq": torch.stack(hqs),
            "lq_path": lq_paths,
            "gt_path": gt_paths,
        }
        if self.has_masks and masks:
            sample["invalid_mask"] = torch.stack(masks)
        return sample


def collate_scene_groups(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate whole exposure families into one flat batch."""
    result = {
        "lq": torch.cat([s["lq"] for s in batch], dim=0),
        "hq": torch.cat([s["hq"] for s in batch], dim=0),
        "lq_path": [p for s in batch for p in s["lq_path"]],
        "gt_path": [p for s in batch for p in s["gt_path"]],
    }
    if "invalid_mask" in batch[0]:
        result["invalid_mask"] = torch.cat([s["invalid_mask"] for s in batch], dim=0)
    return result
