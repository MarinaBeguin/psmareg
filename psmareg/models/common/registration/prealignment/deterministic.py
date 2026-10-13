"""Deterministic low-resolution translation pre-alignment utilities."""

from __future__ import annotations

from typing import Sequence

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from psmareg.models.common.losses import mind_ssc_descriptor
from psmareg.models.common.registration.affine import body_center_translation
from psmareg.models.common.registration.prealignment.similarity import (
    masked_ncc_score,)
from psmareg.models.common.registration.warping import (
    valid_sampling_mask,
    warp_image_3d,
)


def constant_translation_field(
    shape: Sequence[int],
    translation_xyz: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a dense pull-back translation field from [B,3] x/y/z shifts.

    Args:
        shape: Shape ``(B, C, D, H, W)`` of the reference volume.
        translation_xyz: Translations of shape ``[B, 3]``.
        device: Optional device of the field.
        dtype: Optional dtype of the field.

    Returns:
        Field of shape ``[B, 3, D, H, W]``.

    Raises:
        ValueError: If the translation shape does not match the batch.
    """
    batch_size, _, depth, height, width = shape
    if translation_xyz.shape != (batch_size, 3):
        raise ValueError(f"Expected translation shape {(batch_size, 3)}, "
                         f"got {tuple(translation_xyz.shape)}.")
    return translation_xyz.to(device=device, dtype=dtype)[:, :, None, None,
                                                          None].expand(
                                                              -1,
                                                              -1,
                                                              depth,
                                                              height,
                                                              width,
                                                          )


def _masked_feature_l2_score(
    fixed_features: torch.Tensor,
    warped_features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the negative masked mean squared feature distance."""
    if mask.shape[1] == 1:
        mask = mask.expand(-1, fixed_features.shape[1], -1, -1, -1)
    distance = (fixed_features - warped_features).square()
    return -(distance * mask).sum(dim=(1, 2, 3, 4)) / mask.sum(
        dim=(1, 2, 3, 4)).clamp_min(1.0)


@dataclass(frozen=True)
class TranslationPrealignSettings:
    """Resolution, search, and similarity settings for CT pre-alignment."""

    downsample_factor: int = 4
    search_radius: int = 8
    search_step: int = 2
    body_threshold: float = 0.02
    max_translation: float = 80.0
    padding_mode: str = "border"
    feature_mode: str = "ct"


@dataclass(frozen=True)
class _LowresSearchImages:
    """Low-resolution images and masks shared by every translation candidate."""

    fixed: torch.Tensor
    moving: torch.Tensor
    fixed_body: torch.Tensor
    moving_body: torch.Tensor
    fixed_mind: torch.Tensor | None
    moving_mind: torch.Tensor | None


@dataclass(frozen=True)
class _LowresSearchState:
    """Prepared tensors and translation scale for one exhaustive search."""

    images: _LowresSearchImages
    scale_xyz: torch.Tensor
    initial_translation: torch.Tensor
    max_translation: float


def _prepare_lowres_search(
    fixed_ct: torch.Tensor,
    moving_ct: torch.Tensor,
    settings: TranslationPrealignSettings,
) -> _LowresSearchState:
    """Downsample the pair and construct optional MIND features once."""
    _, _, depth, height, width = fixed_ct.shape
    low_size = (
        max(4, int(round(depth / settings.downsample_factor))),
        max(4, int(round(height / settings.downsample_factor))),
        max(4, int(round(width / settings.downsample_factor))),
    )
    fixed_low = F.interpolate(fixed_ct.float(),
                              size=low_size,
                              mode="trilinear",
                              align_corners=False)
    moving_low = F.interpolate(moving_ct.float(),
                               size=low_size,
                               mode="trilinear",
                               align_corners=False)
    low_depth, low_height, low_width = fixed_low.shape[-3:]
    scale_xyz = torch.tensor(
        [width / low_width, height / low_height, depth / low_depth],
        device=fixed_ct.device,
        dtype=fixed_ct.dtype,
    )
    max_translation_low = settings.max_translation / float(
        settings.downsample_factor)
    initial_low = body_center_translation(
        fixed_low,
        moving_low,
        threshold=settings.body_threshold,
        max_translation=max_translation_low,
    ).to(fixed_ct.dtype)
    images = _LowresSearchImages(
        fixed=fixed_low,
        moving=moving_low,
        fixed_body=(fixed_low > settings.body_threshold).float(),
        moving_body=(moving_low > settings.body_threshold).float(),
        fixed_mind=(mind_ssc_descriptor(fixed_low, radius=1, dilation=1)
                    if settings.feature_mode == "mind" else None),
        moving_mind=(mind_ssc_descriptor(moving_low, radius=1, dilation=1)
                     if settings.feature_mode == "mind" else None),
    )
    return _LowresSearchState(images, scale_xyz, initial_low,
                              max_translation_low)


def _score_translation_candidate(
    field: torch.Tensor,
    images: _LowresSearchImages,
    settings: TranslationPrealignSettings,
) -> torch.Tensor:
    """Score one low-resolution translation without changing search order."""
    warped_low = warp_image_3d(images.moving,
                               field,
                               padding_mode=settings.padding_mode)
    warped_body = warp_image_3d(images.moving_body, field,
                                padding_mode="zeros").clamp(0.0, 1.0)
    valid = valid_sampling_mask(field)
    body_union = ((images.fixed_body + warped_body) > 0.0).float()
    mask = (body_union * valid).clamp(0.0, 1.0)

    if images.fixed_mind is not None and images.moving_mind is not None:
        warped_features = warp_image_3d(images.moving_mind,
                                        field,
                                        padding_mode=settings.padding_mode)
        similarity_score = _masked_feature_l2_score(images.fixed_mind,
                                                    warped_features, mask)
    else:
        similarity_score = masked_ncc_score(images.fixed, warped_low, mask)
    intersection = (images.fixed_body * warped_body *
                    valid).sum(dim=(2, 3, 4)).squeeze(1)
    body_dice = ((2.0 * intersection + 1e-6) /
                 ((images.fixed_body * valid).sum(dim=(2, 3, 4)).squeeze(1) +
                  (warped_body * valid).sum(dim=(2, 3, 4)).squeeze(1) + 1e-6))
    return similarity_score + 0.25 * body_dice


@torch.no_grad()
def estimate_lowres_ct_translation_field(
    fixed_ct: torch.Tensor,
    moving_ct: torch.Tensor,
    settings: TranslationPrealignSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate a robust CT translation field by low-resolution grid search.

    The returned displacement follows the repository pull-back convention:
    ``warped(x) = moving(x + displacement(x))``. Translation components are in
    full-resolution voxels ordered as x/y/z.

    Args:
        fixed_ct: Fixed CT of shape ``[B, C, D, H, W]``.
        moving_ct: Moving CT with the same shape.
        settings: Search grid, bounds, and similarity feature.

    Returns:
        Full-resolution translation field and its ``[B, 3]`` translation in
        voxels.

    Raises:
        ValueError: If the volumes differ in shape or a search setting is out of
            range.
    """
    downsample_factor = settings.downsample_factor
    search_radius = settings.search_radius
    search_step = settings.search_step
    max_translation = settings.max_translation
    feature_mode = settings.feature_mode
    if fixed_ct.shape != moving_ct.shape or fixed_ct.ndim != 5:
        raise ValueError("fixed_ct and moving_ct must share shape [B,C,D,H,W].")
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1.")
    if search_step < 1:
        raise ValueError("search_step must be >= 1.")
    if search_radius < 0:
        raise ValueError("search_radius cannot be negative.")
    if feature_mode not in {"ct", "mind"}:
        raise ValueError("feature_mode must be 'ct' or 'mind'.")

    state = _prepare_lowres_search(fixed_ct, moving_ct, settings)
    images = state.images
    batch_size = fixed_ct.shape[0]
    initial_low = state.initial_translation
    max_translation_low = state.max_translation
    best_translation_low = initial_low.clone()
    best_score = torch.full(
        (batch_size,),
        -float("inf"),
        device=fixed_ct.device,
        dtype=fixed_ct.dtype,
    )

    offsets = range(-search_radius, search_radius + 1, search_step)
    for offset_x, offset_y, offset_z in itertools.product(
            offsets, offsets, offsets):
        offset = torch.tensor(
            [offset_x, offset_y, offset_z],
            device=fixed_ct.device,
            dtype=fixed_ct.dtype,
        )
        candidate_low = (initial_low + offset).clamp(
            -max_translation_low,
            max_translation_low,
        )
        candidate_field = constant_translation_field(
            tuple(images.fixed.shape),
            candidate_low,
            device=fixed_ct.device,
            dtype=fixed_ct.dtype,
        )
        score = _score_translation_candidate(candidate_field, images, settings)
        improved = score > best_score
        best_score = torch.where(improved, score, best_score)
        best_translation_low = torch.where(
            improved[:, None],
            candidate_low,
            best_translation_low,
        )

    best_translation_full = (best_translation_low *
                             state.scale_xyz[None, :]).clamp(
                                 -max_translation,
                                 max_translation,
                             )
    field = constant_translation_field(
        tuple(fixed_ct.shape),
        best_translation_full,
        device=fixed_ct.device,
        dtype=fixed_ct.dtype,
    )
    return field, best_translation_full
