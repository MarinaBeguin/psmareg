"""Registration losses used by the final PSMAReg fine-tuning pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from psmareg.config.fine_tuning_config import PetLossConfig
from psmareg.models.common.losses import (
    image_mse_loss,
    label_group_mask,
    smoothness_loss,
    soft_dice_loss,
)
from psmareg.models.common.registration.displacement import (
    resize_voxel_displacement,)
from psmareg.models.common.registration.warping import warp_image_3d


def valid_overlap_hinge_loss(overlap: torch.Tensor,
                             reference_overlap: torch.Tensor,
                             minimum_overlap: float) -> torch.Tensor:
    """Prevent learned fields from discarding avoidable source coverage.

    Args:
        overlap: Valid sampling mask after registration.
        reference_overlap: Valid mask before learned deformation.
        minimum_overlap: Required overlap, capped by reference coverage.

    Returns:
        Mean hinge penalty across batch items.
    """
    overlap_fraction = overlap.mean(dim=(1, 2, 3, 4))
    reference_fraction = reference_overlap.mean(dim=(1, 2, 3, 4)).detach()
    target = torch.minimum(
        reference_fraction,
        overlap_fraction.new_full(overlap_fraction.shape, minimum_overlap))
    return F.relu(target - overlap_fraction).mean()


def downsample_image_for_metric(image: torch.Tensor,
                                factor: int = 4) -> torch.Tensor:
    """Downsample image tensors for logging-only metrics.

    Args:
        image: Input tensor of shape ``[B, C, D, H, W]``.
        factor: Integer spatial reduction factor.

    Returns:
        The input or its trilinearly downsampled version.
    """
    if factor <= 1:
        return image
    target_size = tuple(
        (max(1, int(size // factor)) for size in image.shape[-3:]))
    return F.interpolate(image,
                         size=target_size,
                         mode="trilinear",
                         align_corners=False)


def displacement_smoothness_loss(displacement: torch.Tensor,
                                 downsample_factor: int = 1) -> torch.Tensor:
    """Compute memory-aware first-order displacement smoothness.

    Args:
        displacement: Dense field of shape ``[B, 3, D, H, W]``.
        downsample_factor: Spatial reduction factor for the loss.

    Returns:
        Smoothness penalty adjusted for the coarser voxel scale.
    """
    if downsample_factor <= 1:
        return smoothness_loss(displacement)
    target_size = tuple((max(1, int(size // downsample_factor))
                         for size in displacement.shape[-3:]))
    displacement_low = F.interpolate(displacement,
                                     size=target_size,
                                     mode="trilinear",
                                     align_corners=False)
    return smoothness_loss(displacement_low) / float(downsample_factor**2)


def multiclass_label_dice_loss(fixed_label: torch.Tensor,
                               moving_label: torch.Tensor,
                               displacement: torch.Tensor,
                               num_classes: int,
                               eps: float = 1e-06,
                               padding_mode: str = "border") -> torch.Tensor:
    """Compute differentiable class Dice after warping moving labels.

    Args:
        fixed_label: Fixed integer labels of shape ``[B, D, H, W]``.
        moving_label: Moving labels with the same shape.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        num_classes: Number of label indices, including background.
        eps: Dice denominator stabilizer.
        padding_mode: Padding used while warping masks.

    Returns:
        Mean foreground-class Dice loss, or zero when no class is present.
    """
    losses: list[torch.Tensor] = []
    for class_idx in range(1, num_classes):
        fixed_mask = (fixed_label == class_idx).float().unsqueeze(1)
        moving_mask = (moving_label == class_idx).float().unsqueeze(1)
        if fixed_mask.sum() < eps and moving_mask.sum() < eps:
            continue
        warped_moving_mask = warp_image_3d(moving_mask,
                                           displacement,
                                           interpolation_mode="bilinear",
                                           padding_mode=padding_mode)
        losses.append(soft_dice_loss(warped_moving_mask, fixed_mask, eps))
    if not losses:
        return torch.zeros((), device=displacement.device)
    return torch.stack(losses).mean()


def macro_label_dice_loss(fixed_label: torch.Tensor,
                          moving_label: torch.Tensor,
                          displacement: torch.Tensor,
                          groups: list[tuple[str, tuple[int, ...]]],
                          eps: float = 1e-06,
                          padding_mode: str = "border") -> torch.Tensor:
    """Compute mean Dice loss over stable macro-anatomical groups.

    Args:
        fixed_label: Fixed integer labels of shape ``[B, D, H, W]``.
        moving_label: Moving labels with the same shape.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        groups: Named groups of anatomical label indices.
        eps: Dice denominator stabilizer.
        padding_mode: Padding used while warping masks.

    Returns:
        Mean Dice loss over nonempty groups, or zero if none are present.
    """
    losses: list[torch.Tensor] = []
    for _, class_ids in groups:
        fixed_mask = label_group_mask(fixed_label, class_ids).unsqueeze(1)
        moving_mask = label_group_mask(moving_label, class_ids).unsqueeze(1)
        if fixed_mask.sum() < eps and moving_mask.sum() < eps:
            continue
        warped_moving_mask = warp_image_3d(moving_mask,
                                           displacement,
                                           interpolation_mode="bilinear",
                                           padding_mode=padding_mode)
        losses.append(soft_dice_loss(warped_moving_mask, fixed_mask, eps))
    if not losses:
        return torch.zeros((), device=displacement.device)
    return torch.stack(losses).mean()


def downsample_labels_and_displacement(
        fixed_label: torch.Tensor, moving_label: torch.Tensor,
        displacement: torch.Tensor,
        factor: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downsample paired labels and consistently rescale displacement.

    Args:
        fixed_label: Fixed labels of shape ``[B, D, H, W]``.
        moving_label: Moving labels with the same shape.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        factor: Integer spatial reduction factor.

    Returns:
        Downsampled fixed labels, moving labels, and voxel displacement.
    """
    if factor <= 1:
        return (fixed_label, moving_label, displacement)
    fixed_low = F.interpolate(fixed_label.unsqueeze(1).float(),
                              scale_factor=1.0 / factor,
                              mode="nearest",
                              recompute_scale_factor=False).squeeze(1).long()
    moving_low = F.interpolate(moving_label.unsqueeze(1).float(),
                               scale_factor=1.0 / factor,
                               mode="nearest",
                               recompute_scale_factor=False).squeeze(1).long()
    disp_low = resize_voxel_displacement(displacement,
                                         fixed_low.shape[-3:],
                                         align_corners=False)
    return (fixed_low, moving_low, disp_low)


def _smooth_pet_for_hot_mask(image: torch.Tensor,
                             kernel_size: int) -> torch.Tensor:
    """Smooth PET intensities before estimating high-uptake masks."""
    if kernel_size <= 1:
        return image
    padding = kernel_size // 2
    return F.avg_pool3d(image,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding)


def capped_pet_mse_loss(warped_pet: torch.Tensor,
                        fixed_pet: torch.Tensor,
                        valid_mask: torch.Tensor | None,
                        upper_quantile: float,
                        quantile_downsample: int = 1) -> torch.Tensor:
    """PET MSE that can ignore extreme uptake outliers independently per pair.

    This keeps a weak PET alignment signal while reducing the chance that
    bladder, bowel, or non-corresponding very-hot lesions dominate the
    deformation.

    Args:
        warped_pet: Warped moving PET of shape ``[B, 1, D, H, W]``.
        fixed_pet: Fixed PET with the same shape.
        valid_mask: Optional mask of valid sampled voxels.
        upper_quantile: Upper uptake quantile retained in the loss.
        quantile_downsample: Spatial reduction for estimating quantiles.

    Returns:
        Scalar PET mean-squared-error loss.
    """
    if upper_quantile >= 1.0:
        if valid_mask is None:
            return image_mse_loss(warped_pet, fixed_pet)
        return ((warped_pet - fixed_pet).square() *
                valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    output_shape = fixed_pet.shape[-3:]
    fixed_for_quantile = fixed_pet
    warped_for_quantile = warped_pet
    valid_for_quantile = valid_mask
    if quantile_downsample > 1:
        low_shape = tuple(
            (max(1, size // quantile_downsample) for size in output_shape))
        fixed_for_quantile = F.interpolate(fixed_pet.float(),
                                           size=low_shape,
                                           mode="trilinear",
                                           align_corners=False)
        warped_for_quantile = F.interpolate(warped_pet.float(),
                                            size=low_shape,
                                            mode="trilinear",
                                            align_corners=False)
        if valid_mask is not None:
            valid_for_quantile = F.interpolate(valid_mask.float(),
                                               size=low_shape,
                                               mode="nearest")
    mask = torch.ones_like(fixed_for_quantile[:, :1], dtype=torch.bool)
    if valid_for_quantile is not None:
        mask = valid_for_quantile > 0.5
    for batch_index in range(fixed_for_quantile.shape[0]):
        valid = mask[batch_index:batch_index + 1]
        if int(valid.sum().item()) < 8:
            continue
        fixed_values = fixed_for_quantile[batch_index:batch_index + 1][valid]
        warped_values = warped_for_quantile[batch_index:batch_index + 1][valid]
        fixed_threshold = torch.quantile(fixed_values.float(), upper_quantile)
        warped_threshold = torch.quantile(warped_values.float(), upper_quantile)
        mask[batch_index:batch_index + 1] = valid & (
            fixed_for_quantile[batch_index:batch_index + 1]
            <= fixed_threshold) & (warped_for_quantile[batch_index:batch_index +
                                                       1] <= warped_threshold)
    if quantile_downsample > 1:
        mask = F.interpolate(mask.float(), size=output_shape,
                             mode="nearest") > 0.5
        if valid_mask is not None:
            mask = mask & (valid_mask > 0.5)
    weight = mask.to(dtype=fixed_pet.dtype)
    return ((warped_pet - fixed_pet).square() *
            weight).sum() / weight.sum().clamp_min(1.0)


def pet_hot_region_mse_loss(fixed_pet: torch.Tensor, warped_pet: torch.Tensor,
                            valid_mask: torch.Tensor | None, quantile: float,
                            upper_quantile: float, warped_weight: float,
                            smooth_kernel: int) -> torch.Tensor:
    """Compute a conservative PET loss over robust high-uptake regions.

    Args:
        fixed_pet: Fixed PET of shape ``[B, 1, D, H, W]``.
        warped_pet: Warped moving PET with the same shape.
        valid_mask: Optional valid sampling mask.
        quantile: Lower uptake quantile defining hot regions.
        upper_quantile: Upper uptake quantile excluding extreme outliers.
        warped_weight: Relative contribution of warped PET hot voxels.
        smooth_kernel: Kernel size used before estimating hot regions.

    Returns:
        Weighted PET mean-squared-error loss.
    """
    fixed_smooth = _smooth_pet_for_hot_mask(fixed_pet.float(), smooth_kernel)
    warped_smooth = _smooth_pet_for_hot_mask(warped_pet.float(), smooth_kernel)
    weight = torch.zeros_like(fixed_pet)
    for batch_index in range(fixed_pet.shape[0]):
        valid = valid_mask[
            batch_index:batch_index +
            1] > 0.5 if valid_mask is not None else torch.ones_like(
                fixed_pet[batch_index:batch_index + 1], dtype=torch.bool)
        if valid.sum() == 0:
            continue
        fixed_values = fixed_smooth[batch_index:batch_index + 1][valid]
        warped_values = warped_smooth[batch_index:batch_index + 1][valid]
        fixed_low = torch.quantile(fixed_values, quantile)
        warped_low = torch.quantile(warped_values, quantile)
        fixed_mask = fixed_smooth[batch_index:batch_index + 1] >= fixed_low
        warped_mask = warped_smooth[batch_index:batch_index + 1] >= warped_low
        if upper_quantile < 1.0:
            fixed_high = torch.quantile(fixed_values, upper_quantile)
            warped_high = torch.quantile(warped_values, upper_quantile)
            fixed_mask = fixed_mask & (fixed_smooth[batch_index:batch_index + 1]
                                       <= fixed_high)
            warped_mask = warped_mask & (warped_smooth[batch_index:batch_index +
                                                       1] <= warped_high)
        batch_weight = fixed_mask.float() + warped_weight * warped_mask.float()
        weight[batch_index:batch_index + 1] = batch_weight * valid.float()
    return ((warped_pet - fixed_pet).square() *
            weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def adaptive_pet_correspondence_weight(
        fixed_pet: torch.Tensor, warped_pet: torch.Tensor,
        valid_mask: torch.Tensor | None,
        settings: PetLossConfig) -> tuple[torch.Tensor, float]:
    """Estimate whether hot PET regions correspond and return a PET loss weight.

    The returned scalar is detached and intentionally non-differentiable. It is
    a safety gate: if longitudinal PET hot regions do not overlap after the
    current warp, PET losses should not strongly pull the deformation toward
    non-shared uptake patterns.

    Args:
        fixed_pet: Fixed PET of shape ``[B, 1, D, H, W]``.
        warped_pet: Warped moving PET with the same shape.
        valid_mask: Optional valid sampling mask.
        settings: Validated adaptive PET correspondence parameters.

    Returns:
        Detached adaptive weight and hot-region Dice score.
    """
    quantile = settings.adaptive_pet_hot_quantile
    smooth_kernel = settings.adaptive_pet_smooth_kernel
    downsample = settings.adaptive_pet_downsample
    min_weight = settings.adaptive_pet_min_weight
    low_dice = settings.adaptive_pet_low_dice
    high_dice = settings.adaptive_pet_high_dice
    if downsample > 1:
        output_size = tuple(
            (max(1, size // downsample) for size in fixed_pet.shape[-3:]))
        fixed_pet = F.interpolate(fixed_pet.float(),
                                  size=output_size,
                                  mode="trilinear",
                                  align_corners=False)
        warped_pet = F.interpolate(warped_pet.float(),
                                   size=output_size,
                                   mode="trilinear",
                                   align_corners=False)
        if valid_mask is not None:
            valid_mask = F.interpolate(valid_mask.float(),
                                       size=output_size,
                                       mode="nearest")
    if smooth_kernel > 1:
        padding = smooth_kernel // 2
        fixed_smooth = F.avg_pool3d(fixed_pet.float(),
                                    kernel_size=smooth_kernel,
                                    stride=1,
                                    padding=padding)
        warped_smooth = F.avg_pool3d(warped_pet.float(),
                                     kernel_size=smooth_kernel,
                                     stride=1,
                                     padding=padding)
    else:
        fixed_smooth = fixed_pet.float()
        warped_smooth = warped_pet.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(fixed_smooth[:, :1], dtype=torch.bool)
    else:
        valid_mask = valid_mask > 0.5
    dices: list[torch.Tensor] = []
    for batch_index in range(fixed_smooth.shape[0]):
        valid = valid_mask[batch_index:batch_index + 1]
        if int(valid.sum().item()) < 8:
            dices.append(torch.zeros((), device=fixed_pet.device))
            continue
        fixed_values = fixed_smooth[batch_index:batch_index + 1][valid]
        warped_values = warped_smooth[batch_index:batch_index + 1][valid]
        fixed_threshold = torch.quantile(fixed_values, quantile)
        warped_threshold = torch.quantile(warped_values, quantile)
        fixed_hot = (fixed_smooth[batch_index:batch_index + 1]
                     >= fixed_threshold) & valid
        warped_hot = (warped_smooth[batch_index:batch_index + 1]
                      >= warped_threshold) & valid
        intersection = (fixed_hot & warped_hot).float().sum()
        denominator = fixed_hot.float().sum() + warped_hot.float().sum()
        dice = (2.0 * intersection + 1e-06) / (denominator + 1e-06)
        dices.append(dice)
    correspondence = torch.stack(dices).mean()
    progress = ((correspondence - low_dice) / (high_dice - low_dice)).clamp(
        0.0, 1.0)
    min_weight_tensor = torch.as_tensor(min_weight,
                                        device=fixed_pet.device,
                                        dtype=fixed_pet.dtype).clamp(0.0, 1.0)
    weight = min_weight_tensor + (1.0 - min_weight_tensor) * progress.to(
        fixed_pet.dtype)
    return (weight, float(correspondence.detach().cpu()))
