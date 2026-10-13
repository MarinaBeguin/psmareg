"""Challenge-like validation metrics used during checkpoint selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from psmareg.models.common.registration.warping import warp_image_3d


def compute_multiclass_dsc_score(fixed_label: torch.Tensor,
                                 moving_label: torch.Tensor,
                                 displacement: torch.Tensor,
                                 num_classes: int,
                                 eps: float = 1e-06,
                                 padding_mode: str = "border") -> float:
    """Compute mean multi-class Dice score after warping moving labels.

    Args:
        fixed_label: Fixed labels of shape ``[B, D, H, W]``.
        moving_label: Moving labels with the same shape.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        num_classes: Number of label classes, background included.
        eps: Numerical stabilizer of the Dice ratio.
        padding_mode: Padding used when warping the moving labels.

    Returns:
        Mean Dice over the classes present in either label map, or NaN when
        there is none.
    """
    warped_label = warp_image_3d(moving_label.unsqueeze(1).float(),
                                 displacement,
                                 interpolation_mode="nearest",
                                 padding_mode=padding_mode).squeeze(1).long()
    warped_scores: list[float] = []
    for class_idx in range(1, num_classes):
        fixed_mask = fixed_label == class_idx
        warped_mask = warped_label == class_idx
        if fixed_mask.sum() < eps and warped_mask.sum() < eps:
            continue
        intersection = torch.logical_and(warped_mask, fixed_mask).sum().float()
        denominator = warped_mask.sum().float() + fixed_mask.sum().float()
        if float(denominator.detach().cpu()) == 0.0:
            warped_scores.append(1.0)
        else:
            dice = (2.0 * intersection + eps) / (denominator + eps)
            warped_scores.append(float(dice.detach().cpu()))
    if not warped_scores:
        return float("nan")
    return sum(warped_scores) / len(warped_scores)


def _challenge_like_sampling_grid(displacement: torch.Tensor) -> torch.Tensor:
    """Build the normalized sampling grid used by challenge-like warps.

    Internal fields use grid-sample component order [W, H, D]. The exported
    NIfTI stores vector components in array-axis order [D, H, W], then Codabench
    upsamples and feeds that axis-order flow to its SpatialTransformer. This
    helper mimics that whole path without touching disk.
    """
    low_res_axis_flow = F.interpolate(displacement[:, [2, 1, 0]],
                                      scale_factor=0.5,
                                      mode="trilinear",
                                      align_corners=False,
                                      recompute_scale_factor=False)
    axis_flow = F.interpolate(low_res_axis_flow,
                              scale_factor=2.0,
                              mode="trilinear",
                              align_corners=False,
                              recompute_scale_factor=False)
    _, _, depth, height, width = axis_flow.shape
    vectors = [
        torch.arange(0, size, device=axis_flow.device, dtype=axis_flow.dtype)
        for size in (depth, height, width)
    ]
    grids = torch.meshgrid(*vectors, indexing="ij")
    grid = torch.stack(grids, dim=0).unsqueeze(0)
    new_locs = grid + axis_flow
    shape = axis_flow.shape[2:]
    for axis, size in enumerate(shape):
        new_locs[:, axis] = 2.0 * (new_locs[:, axis] / max(size - 1, 1) - 0.5)
    return new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]


def _challenge_like_warp_image(
        image: torch.Tensor,
        displacement: torch.Tensor,
        interpolation_mode: str = "bilinear") -> torch.Tensor:
    """Warp an image using the same export/upsample assumptions as Codabench.

    The official scorer upsamples submitted displacement fields with
    ``align_corners=False`` but then applies them with its VoxelMorph-style
    SpatialTransformer, which normalizes coordinates as ``2*x/(N-1)-1`` and
    calls ``grid_sample(..., align_corners=True)``.  Keep this distinction
    explicit: changing the warp to ``align_corners=False`` creates small but
    visible MTV/TLG bias even for identity/rigid fields.
    """
    return F.grid_sample(image.float(),
                         _challenge_like_sampling_grid(displacement),
                         mode=interpolation_mode,
                         padding_mode="zeros",
                         align_corners=True)


def _challenge_like_warp_label_map(label: torch.Tensor,
                                   displacement: torch.Tensor) -> torch.Tensor:
    """Warp a label map like the Codabench scorer after submission export."""
    warped = _challenge_like_warp_image(label.float().unsqueeze(1),
                                        displacement,
                                        interpolation_mode="nearest")
    return torch.round(warped.squeeze(1)).long()


def compute_challenge_like_ct_dsc_score(fixed_label: torch.Tensor,
                                        moving_label: torch.Tensor,
                                        displacement: torch.Tensor,
                                        num_classes: int,
                                        eps: float = 1e-06) -> float:
    """Compute CT-label Dice with the Codabench warp and export assumptions.

    Args:
        fixed_label: Fixed labels of shape ``[B, D, H, W]``.
        moving_label: Moving labels with the same shape.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        num_classes: Number of label classes, background included.
        eps: Numerical stabilizer of the Dice ratio.

    Returns:
        Mean Dice over the classes present, or NaN when there is none.
    """
    warped_label = _challenge_like_warp_label_map(moving_label, displacement)
    scores: list[float] = []
    for class_idx in range(1, num_classes):
        fixed_mask = fixed_label == class_idx
        warped_mask = warped_label == class_idx
        denominator = fixed_mask.sum().float() + warped_mask.sum().float()
        if float(denominator.detach().cpu()) == 0.0:
            continue
        intersection = torch.logical_and(fixed_mask, warped_mask).sum().float()
        dice = (2.0 * intersection + eps) / (denominator + eps)
        scores.append(float(dice.detach().cpu()))
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def compute_challenge_like_binary_pet_scores(fixed_pet_label: torch.Tensor,
                                             moving_pet_label: torch.Tensor,
                                             displacement: torch.Tensor,
                                             eps: float = 1e-06) -> float:
    """Compute binary PET lesion Dice with challenge-like label warping.

    Args:
        fixed_pet_label: Fixed lesion labels of shape ``[B, D, H, W]``.
        moving_pet_label: Moving lesion labels with the same shape.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        eps: Numerical stabilizer of the Dice ratio.

    Returns:
        Dice of the binary lesion masks.
    """
    warped_pet_label = _challenge_like_warp_label_map(moving_pet_label,
                                                      displacement)
    fixed_mask = fixed_pet_label > 0
    warped_mask = warped_pet_label > 0
    denominator = fixed_mask.sum().float() + warped_mask.sum().float()
    if float(denominator.detach().cpu()) == 0.0:
        pet_dsc = 1.0
    else:
        intersection = torch.logical_and(fixed_mask, warped_mask).sum().float()
        pet_dsc = float(
            ((2.0 * intersection + eps) / (denominator + eps)).detach().cpu())
    return pet_dsc


def compute_challenge_like_pet_volume_errors(
        moving_pet: torch.Tensor,
        moving_pet_label: torch.Tensor,
        displacement: torch.Tensor,
        eps: float = 1e-06) -> tuple[float, float]:
    """Approximate Codabench MTV/TLG percent errors on normalized PET.

    The official scorer uses raw moving PET intensities. During training we only
    have the normalized tensor, so MTV is exact and TLG is a proxy rather than a
    bit-for-bit reproduction. The proxy is still spatially consistent: both the
    PET image and the PET label are warped through the same challenge-like path.

    Args:
        moving_pet: Moving PET of shape ``[B, 1, D, H, W]``.
        moving_pet_label: Moving lesion labels of shape ``[B, D, H, W]``.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        eps: Numerical stabilizer of the relative errors.

    Returns:
        Percent errors of the metabolic tumor volume and of the total lesion
        uptake after warping.
    """
    warped_pet_label = _challenge_like_warp_label_map(moving_pet_label,
                                                      displacement)
    warped_pet = _challenge_like_warp_image(moving_pet,
                                            displacement,
                                            interpolation_mode="bilinear")
    before_mask = moving_pet_label > 0
    after_mask = warped_pet_label > 0
    mtv_before = before_mask.sum().float()
    mtv_after = after_mask.sum().float()
    tlg_before = (before_mask.unsqueeze(1).float() * moving_pet).sum()
    tlg_after = (after_mask.unsqueeze(1).float() * warped_pet).sum()
    mtv_percent_error = 100.0 * (mtv_after -
                                 mtv_before).abs() / mtv_before.clamp_min(eps)
    tlg_percent_error = 100.0 * (tlg_after -
                                 tlg_before).abs() / tlg_before.clamp_min(eps)
    return (float(mtv_percent_error.detach().cpu()),
            float(tlg_percent_error.detach().cpu()))
