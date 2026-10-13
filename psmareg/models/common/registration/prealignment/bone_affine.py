"""Bone-aware rigid/affine pre-alignment for PSMAReg inference.

The registration network is already good at the deformable part. This module is
for the remaining hard cases where dense deformation is asked to compensate a
global skeletal mismatch. It estimates a small low-resolution affine pull-back
field from CT only, with a conservative fallback when the low-res score does not
improve.
"""

from __future__ import annotations

from typing import Sequence

from dataclasses import dataclass
import functools

import torch
import torch.nn.functional as F

from psmareg.models.common.losses import mind_ssc_descriptor
from psmareg.models.common.registration.affine import body_center_translation
from psmareg.models.common.registration.geometry import (
    euler_rotation_matrix_xyz,
    voxel_grid_xyz,
)
from psmareg.models.common.registration.prealignment.similarity import (
    masked_ncc_score,)
from psmareg.models.common.registration.warping import (
    valid_sampling_mask,
    warp_image_3d,
)


@dataclass(frozen=True)
class BoneAffinePrealignInfo:
    """Diagnostics returned by the bone affine pre-aligner."""

    accepted: bool
    baseline_score: float
    affine_score: float
    improvement: float
    translation_xyz: tuple[float, float, float]
    rotation_xyz_degrees: tuple[float, float, float]
    scale_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class BoneAffineSettings:
    """Bounds and optimization settings for test-time bone alignment."""

    downsample_factors: tuple[int, ...] = (8, 4)
    iterations_per_level: int = 80
    lr: float = 0.05
    bone_threshold: float = 0.66
    bone_softness: float = 0.04
    max_translation: float = 120.0
    max_rotation_degrees: float = 8.0
    max_scaling: float = 0.04
    min_improvement: float = 0.005
    padding_mode: str = "border"


@dataclass(frozen=True)
class _BoneLevel:
    """CT, bone, and MIND tensors prepared at one optimization scale."""

    fixed_ct: torch.Tensor
    moving_ct: torch.Tensor
    fixed_bone: torch.Tensor
    moving_bone: torch.Tensor
    fixed_mind: torch.Tensor
    moving_mind: torch.Tensor


def _downsample(image: torch.Tensor, factor: int) -> torch.Tensor:
    """Resample an image by ``factor``, keeping at least four voxels."""
    if factor <= 1:
        return image.float()
    size = tuple(max(4, int(round(dim / factor))) for dim in image.shape[-3:])
    return F.interpolate(image.float(),
                         size=size,
                         mode="trilinear",
                         align_corners=False)


def _soft_bone(ct: torch.Tensor, threshold: float,
               softness: float) -> torch.Tensor:
    """Return a soft, slightly dilated bone probability map."""
    bone = torch.sigmoid((ct.float() - threshold) / max(softness, 1e-4))
    # A tiny dilation makes the metric less brittle to one-voxel cortical edges.
    return F.max_pool3d(bone, kernel_size=3, stride=1,
                        padding=1).clamp(0.0, 1.0)


def _body_mask(ct: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """Threshold the normalized CT into a body mask."""
    return (ct > threshold).float()


def bone_affine_parameters_to_displacement(
    reference_ct: torch.Tensor,
    *,
    translation_full_xyz: torch.Tensor,
    rotation_xyz_radians: torch.Tensor,
    log_scale_xyz: torch.Tensor,
    full_shape: Sequence[int],
    settings: BoneAffineSettings,
) -> torch.Tensor:
    """Create a dense pull-back field from bounded affine parameters.

    Args:
        reference_ct: Volume defining the grid, device, and dtype of the field.
        translation_full_xyz: Unbounded translation parameters of shape ``[1,
            3]``, expressed at full resolution.
        rotation_xyz_radians: Unbounded rotation parameters of shape ``[1, 3]``.
        log_scale_xyz: Unbounded scale parameters of shape ``[1, 3]``.
        full_shape: Spatial shape of the full-resolution volume.
        settings: Bounds applied to the three parameter groups.

    Returns:
        Field of shape ``[B, 3, D, H, W]`` on the grid of ``reference_ct``.
    """
    batch_size, _, depth, height, width = reference_ct.shape
    full_depth, full_height, full_width = full_shape
    device = reference_ct.device
    dtype = reference_ct.dtype
    scale_to_current = torch.tensor(
        [
            width / max(full_width, 1),
            height / max(full_height, 1),
            depth / max(full_depth, 1),
        ],
        device=device,
        dtype=dtype,
    )
    translation = settings.max_translation * torch.tanh(
        translation_full_xyz) * scale_to_current[None]
    max_angle = torch.tensor(
        settings.max_rotation_degrees * torch.pi / 180.0,
        device=device,
        dtype=dtype,
    )
    angles = max_angle * torch.tanh(rotation_xyz_radians)
    scale = 1.0 + settings.max_scaling * torch.tanh(log_scale_xyz)

    rotation = euler_rotation_matrix_xyz(angles)
    affine = rotation @ torch.diag_embed(scale)
    grid = voxel_grid_xyz((depth, height, width), device=device,
                          dtype=dtype)[None].expand(batch_size, -1, -1, -1, -1)
    center = torch.tensor(
        [(width - 1) * 0.5, (height - 1) * 0.5, (depth - 1) * 0.5],
        device=device,
        dtype=dtype,
    )
    flat = grid.flatten(2).transpose(1, 2)
    source = (flat - center[None, None]) @ affine.transpose(1, 2) + center[None,
                                                                           None]
    source = source + translation[:, None]
    displacement = (source - flat).transpose(1,
                                             2).reshape(batch_size, 3, depth,
                                                        height, width)
    return displacement


def _bone_dice_score(fixed_bone: torch.Tensor, moving_bone: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Compute the masked soft Dice of two bone maps."""
    intersection = (fixed_bone * moving_bone * mask).sum(dim=(2, 3,
                                                              4)).squeeze(1)
    denominator = ((fixed_bone * mask).sum(dim=(2, 3, 4)).squeeze(1) +
                   (moving_bone * mask).sum(dim=(2, 3, 4)).squeeze(1))
    return (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def _score_alignment(
    fixed_ct: torch.Tensor,
    moving_ct: torch.Tensor,
    fixed_bone: torch.Tensor,
    moving_bone: torch.Tensor,
    displacement: torch.Tensor,
    *,
    padding_mode: str,
) -> torch.Tensor:
    """Score a field by CT correlation, bone overlap, and CT error."""
    warped_ct = warp_image_3d(moving_ct,
                              displacement,
                              padding_mode=padding_mode)
    warped_bone = warp_image_3d(moving_bone, displacement,
                                padding_mode="zeros").clamp(0.0, 1.0)
    valid = valid_sampling_mask(displacement)
    body = ((_body_mask(fixed_ct) + warp_image_3d(
        _body_mask(moving_ct), displacement, padding_mode="zeros"))
            > 0).float()
    mask = (valid * body).clamp(0.0, 1.0)
    return (0.45 * masked_ncc_score(fixed_ct, warped_ct, mask) +
            0.45 * _bone_dice_score(fixed_bone, warped_bone, mask) - 0.10 *
            (warped_ct - fixed_ct).abs().mul(mask).sum(dim=(1, 2, 3, 4)) /
            mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0))


def _loss_alignment(
    level: _BoneLevel,
    displacement: torch.Tensor,
    *,
    padding_mode: str,
) -> torch.Tensor:
    """Combine the MIND, bone, and CT terms optimized at one level."""
    warped_ct = warp_image_3d(level.moving_ct,
                              displacement,
                              padding_mode=padding_mode)
    warped_bone = warp_image_3d(level.moving_bone,
                                displacement,
                                padding_mode="zeros").clamp(0.0, 1.0)
    warped_mind = warp_image_3d(level.moving_mind,
                                displacement,
                                padding_mode=padding_mode)
    valid = valid_sampling_mask(displacement)
    body = ((_body_mask(level.fixed_ct) + warp_image_3d(
        _body_mask(level.moving_ct), displacement, padding_mode="zeros"))
            > 0).float()
    mask = (valid * body).clamp(0.0, 1.0)
    if mask.shape[1] == 1:
        mind_mask = mask.expand(-1, level.fixed_mind.shape[1], -1, -1, -1)
    else:
        mind_mask = mask
    mind_loss = (level.fixed_mind - warped_mind
                ).abs().mul(mind_mask).sum() / mind_mask.sum().clamp_min(1.0)
    bone_loss = 1.0 - _bone_dice_score(level.fixed_bone, warped_bone,
                                       mask).mean()
    ct_loss = -(0.5 +
                0.5 * masked_ncc_score(level.fixed_ct, warped_ct, mask)).mean()
    return 0.45 * mind_loss + 0.35 * bone_loss + 0.20 * ct_loss


@torch.no_grad()
def _initial_translation_full(
    fixed_low: torch.Tensor,
    moving_low: torch.Tensor,
    *,
    full_shape: Sequence[int],
    max_translation: float,
) -> torch.Tensor:
    """Estimate the body-center translation in full-resolution voxels."""
    low_translation = body_center_translation(
        fixed_low,
        moving_low,
        threshold=0.02,
        max_translation=max_translation,
    )
    low_depth, low_height, low_width = fixed_low.shape[-3:]
    full_depth, full_height, full_width = full_shape
    scale_to_full = torch.tensor(
        [
            full_width / max(low_width, 1),
            full_height / max(low_height, 1),
            full_depth / max(low_depth, 1),
        ],
        device=fixed_low.device,
        dtype=fixed_low.dtype,
    )
    return (low_translation * scale_to_full[None]).clamp(
        -max_translation, max_translation)


@dataclass(frozen=True)
class _AffineFit:
    """Inputs and unbounded affine parameters of one fitted alignment."""

    fixed_ct: torch.Tensor
    moving_ct: torch.Tensor
    factors: tuple[int, ...]
    translation: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor


class BoneAffineAligner:
    """Fit and evaluate a conservative test-time CT/bone affine transform."""

    def __init__(self, settings: BoneAffineSettings | None = None) -> None:
        """Create an aligner without fitting it.

        Args:
            settings: Optional optimization and acceptance settings.
        """
        self.settings = settings or BoneAffineSettings()
        self._fit: _AffineFit | None = None

    def fit(self, fixed_ct: torch.Tensor,
            moving_ct: torch.Tensor) -> BoneAffineAligner:
        """Optimize translation, rotation, and scale at successive levels.

        Args:
            fixed_ct: Fixed CT with shape ``[1, C, D, H, W]``.
            moving_ct: Moving CT with the same shape.

        Returns:
            This fitted aligner, ready for ``evaluate``.

        Raises:
            ValueError: If the volumes differ in shape or hold more than one
                pair.
        """
        if fixed_ct.shape != moving_ct.shape or fixed_ct.ndim != 5:
            raise ValueError(
                "fixed_ct and moving_ct must share shape [B,C,D,H,W].")
        if fixed_ct.shape[0] != 1:
            raise ValueError(
                "Bone affine prealign currently expects batch size 1.")
        self._fit = None
        settings = self.settings
        factors = tuple(
            sorted(set(
                max(1, int(value)) for value in settings.downsample_factors),
                   reverse=True))
        initial_translation = _initial_translation_full(
            _downsample(fixed_ct, factors[0]),
            _downsample(moving_ct, factors[0]),
            full_shape=fixed_ct.shape[-3:],
            max_translation=settings.max_translation,
        )
        # The visualizer may call this under torch.no_grad(). Re-enable
        # gradients only for the local affine optimization.
        with torch.enable_grad():
            initial_ratio = (initial_translation /
                             settings.max_translation).clamp(-0.95, 0.95)
            fit = _AffineFit(
                fixed_ct=fixed_ct,
                moving_ct=moving_ct,
                factors=factors,
                translation=torch.atanh(
                    initial_ratio).detach().clone().requires_grad_(True),
                rotation=torch.zeros((1, 3),
                                     device=fixed_ct.device,
                                     dtype=fixed_ct.dtype,
                                     requires_grad=True),
                scale=torch.zeros((1, 3),
                                  device=fixed_ct.device,
                                  dtype=fixed_ct.dtype,
                                  requires_grad=True),
            )
            for factor in factors:
                self._optimize_level(fit, factor)
        self._fit = fit
        return self

    def _field(self,
               fit: _AffineFit,
               reference_ct: torch.Tensor,
               detach: bool = False) -> torch.Tensor:
        """Build the affine field on the grid of ``reference_ct``."""
        translation, rotation, scale = (fit.translation, fit.rotation,
                                        fit.scale)
        if detach:
            translation, rotation, scale = (translation.detach(),
                                            rotation.detach(), scale.detach())
        return bone_affine_parameters_to_displacement(
            reference_ct,
            translation_full_xyz=translation,
            rotation_xyz_radians=rotation,
            log_scale_xyz=scale,
            full_shape=fit.fixed_ct.shape[-3:],
            settings=self.settings,
        )

    def _bones(self, fixed_ct: torch.Tensor,
               moving_ct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the soft bone maps of both CT volumes."""
        settings = self.settings
        return (_soft_bone(fixed_ct, settings.bone_threshold,
                           settings.bone_softness),
                _soft_bone(moving_ct, settings.bone_threshold,
                           settings.bone_softness))

    def _optimize_level(self, fit: _AffineFit, factor: int) -> None:
        """Run the Adam iterations of one resolution level."""
        settings = self.settings
        fixed_low = _downsample(fit.fixed_ct, factor).detach()
        moving_low = _downsample(fit.moving_ct, factor).detach()
        fixed_bone, moving_bone = self._bones(fixed_low, moving_low)
        level = _BoneLevel(
            fixed_low, moving_low, fixed_bone.detach(), moving_bone.detach(),
            mind_ssc_descriptor(fixed_low, radius=1, dilation=1).detach(),
            mind_ssc_descriptor(moving_low, radius=1, dilation=1).detach())
        optimizer = torch.optim.Adam([fit.translation, fit.rotation, fit.scale],
                                     lr=settings.lr)
        for _ in range(settings.iterations_per_level):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_alignment(level,
                                   self._field(fit, fixed_low),
                                   padding_mode=settings.padding_mode)
            # Weak regularization keeps the affine pre-alignment conservative
            # so the deformable cascade remains responsible for local motion.
            loss = (loss + 0.002 * fit.rotation.square().mean() +
                    0.002 * fit.scale.square().mean() +
                    0.0005 * fit.translation.square().mean())
            loss.backward()
            optimizer.step()

    def evaluate(self) -> tuple[torch.Tensor, BoneAffinePrealignInfo]:
        """Accept or reject the fitted transform against a zero baseline.

        Returns:
            Full-resolution displacement and diagnostic scores.

        Raises:
            RuntimeError: The aligner has not been fitted.
        """
        fit = self._fit
        if fit is None:
            raise RuntimeError("Fit the bone affine aligner before evaluation.")
        settings = self.settings
        fixed_eval = _downsample(fit.fixed_ct, fit.factors[-1])
        moving_eval = _downsample(fit.moving_ct, fit.factors[-1])
        fixed_bone, moving_bone = self._bones(fixed_eval, moving_eval)
        zero_field = torch.zeros(fixed_eval.shape[0],
                                 3,
                                 *fixed_eval.shape[-3:],
                                 device=fit.fixed_ct.device,
                                 dtype=fit.fixed_ct.dtype)
        score = functools.partial(_score_alignment,
                                  fixed_eval,
                                  moving_eval,
                                  fixed_bone,
                                  moving_bone,
                                  padding_mode=settings.padding_mode)
        baseline_score = score(zero_field)
        affine_score = score(self._field(fit, fixed_eval))
        improvement = float((affine_score - baseline_score).detach().cpu()[0])
        accepted = improvement >= settings.min_improvement

        with torch.no_grad():
            full_field = self._field(fit, fit.fixed_ct, detach=True)
            if not accepted:
                full_field = torch.zeros_like(full_field)
            info = BoneAffinePrealignInfo(
                accepted=accepted,
                baseline_score=float(baseline_score.detach().cpu()[0]),
                affine_score=float(affine_score.detach().cpu()[0]),
                improvement=improvement,
                translation_xyz=_as_triplet(settings.max_translation *
                                            torch.tanh(fit.translation)),
                rotation_xyz_degrees=_as_triplet(settings.max_rotation_degrees *
                                                 torch.tanh(fit.rotation)),
                scale_xyz=_as_triplet(1.0 + settings.max_scaling *
                                      torch.tanh(fit.scale)),
            )
        return full_field, info


def _as_triplet(values: torch.Tensor) -> tuple[float, float, float]:
    """Convert the first row of a ``[1, 3]`` tensor to Python floats."""
    x, y, z = (float(value) for value in values[0].detach().cpu().tolist())
    return x, y, z
