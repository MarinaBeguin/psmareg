"""Loss primitives shared by pretraining and fine-tuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def image_mse_loss(prediction: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error."""
    return F.mse_loss(prediction, target)


def _shift_3d(image: torch.Tensor, offset: tuple[int, int,
                                                 int]) -> torch.Tensor:
    """Shift a ``[B, C, D, H, W]`` tensor with replicate padding.

    Offsets are given in z/y/x voxel order. The output has the same shape as the
    input and is differentiable w.r.t. image intensities. This small helper
    keeps MIND-style descriptors self-contained and avoids grid construction
    overhead.
    """
    offset_z, offset_y, offset_x = offset
    pad = (
        max(offset_x, 0),
        max(-offset_x, 0),
        max(offset_y, 0),
        max(-offset_y, 0),
        max(offset_z, 0),
        max(-offset_z, 0),
    )
    padded = F.pad(image, pad, mode="replicate")
    depth, height, width = image.shape[-3:]
    start_z = max(-offset_z, 0)
    start_y = max(-offset_y, 0)
    start_x = max(-offset_x, 0)
    return padded[
        :,
        :,
        start_z:start_z + depth,
        start_y:start_y + height,
        start_x:start_x + width,
    ]


def mind_ssc_descriptor(
    image: torch.Tensor,
    *,
    radius: int = 1,
    dilation: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a compact modality-independent self-similarity descriptor.

    This is a lightweight MIND-SSC-style descriptor for training losses and
    coarse prealignment. It encodes local self-similarity by comparing each
    voxel to a fixed set of shifted neighbours, then exponentiating distances
    normalized by the local distance variance. The channel count is deliberately
    modest to keep 3D full-volume fine-tuning affordable.

    Args:
        image: Image of shape ``[B, C, D, H, W]``.
        radius: Radius of the patches whose distances are compared.
        dilation: Spacing between the compared neighbors.
        eps: Numerical stabilizer of the variance normalization.

    Returns:
        Descriptor of shape ``[B, 12, D, H, W]``.

    Raises:
        ValueError: If ``image`` does not have five dimensions.
    """
    if image.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(image.shape)}.")
    image = image.float()
    offsets = (
        (dilation, 0, 0),
        (-dilation, 0, 0),
        (0, dilation, 0),
        (0, -dilation, 0),
        (0, 0, dilation),
        (0, 0, -dilation),
        (dilation, dilation, 0),
        (-dilation, -dilation, 0),
        (dilation, 0, dilation),
        (-dilation, 0, -dilation),
        (0, dilation, dilation),
        (0, -dilation, -dilation),
    )
    padding = radius
    kernel_size = 2 * radius + 1
    distances = []
    for offset in offsets:
        shifted = _shift_3d(image, offset)
        distance = (image - shifted).square()
        distance = F.avg_pool3d(
            distance,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        distances.append(distance.mean(dim=1, keepdim=True))
    descriptor_distances = torch.cat(distances, dim=1)
    variance = descriptor_distances.mean(dim=1, keepdim=True).clamp_min(eps)
    descriptor = torch.exp(-descriptor_distances / variance)
    descriptor = descriptor / descriptor.amax(dim=1,
                                              keepdim=True).clamp_min(eps)
    return descriptor


def mind_ssc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    radius: int = 1,
    dilation: int = 2,
) -> torch.Tensor:
    """MIND-SSC descriptor L1 loss for robust mono/multimodal alignment.

    Args:
        moving: Warped moving image of shape ``[B, C, D, H, W]``.
        fixed: Fixed image with the same shape.
        mask: Optional validity mask of shape ``[B, 1, D, H, W]``.
        radius: Patch radius of the descriptor.
        dilation: Neighbor spacing of the descriptor.

    Returns:
        Mean absolute descriptor difference inside the mask.

    Raises:
        ValueError: If the mask shape does not match the images.
    """
    moving_desc = mind_ssc_descriptor(moving, radius=radius, dilation=dilation)
    fixed_desc = mind_ssc_descriptor(fixed, radius=radius, dilation=dilation)
    difference = (moving_desc - fixed_desc).abs()
    if mask is None:
        return difference.mean()
    if mask.ndim != 5:
        raise ValueError("mask must have shape [B,1,D,H,W].")
    mask = mask.float()
    if mask.shape[-3:] != difference.shape[-3:]:
        mask = F.interpolate(mask, size=difference.shape[-3:], mode="nearest")
    mask = mask.expand(-1, difference.shape[1], -1, -1, -1)
    return (difference * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_global_ncc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the negative global NCC inside an optional mask."""
    if mask is None:
        mask = torch.ones_like(moving[:, :1])
    if mask.shape[-3:] != moving.shape[-3:]:
        mask = F.interpolate(mask.float(),
                             size=moving.shape[-3:],
                             mode="nearest")
    mask = mask.float().expand(-1, moving.shape[1], -1, -1, -1)
    mask_sum = mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
    moving_mean = (moving * mask).sum(dim=(2, 3, 4), keepdim=True) / mask_sum
    fixed_mean = (fixed * mask).sum(dim=(2, 3, 4), keepdim=True) / mask_sum
    moving_centered = (moving - moving_mean) * mask
    fixed_centered = (fixed - fixed_mean) * mask
    numerator = (moving_centered * fixed_centered).sum(dim=(2, 3, 4))
    denominator = torch.sqrt(moving_centered.square().sum(dim=(2, 3, 4)) *
                             fixed_centered.square().sum(dim=(2, 3, 4)) + eps)
    return -(numerator / denominator.clamp_min(eps)).mean()


def multiscale_correlation_loss(
        moving: torch.Tensor,
        fixed: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """Global NCC/correlation loss averaged over several image scales.

    Args:
        moving: Warped moving image of shape ``[B, C, D, H, W]``.
        fixed: Fixed image with the same shape.
        mask: Optional validity mask of shape ``[B, 1, D, H, W]``.
        scales: Average-pooling factors at which the correlation is taken.

    Returns:
        Negative correlation averaged over the scales.

    Raises:
        ValueError: If the two images differ in shape.
    """
    if moving.shape != fixed.shape:
        raise ValueError("moving and fixed must have identical shapes.")
    losses = []
    for scale in scales:
        if scale <= 1:
            moving_scale = moving.float()
            fixed_scale = fixed.float()
            mask_scale = mask
        else:
            moving_scale = F.avg_pool3d(
                moving.float(),
                kernel_size=scale,
                stride=scale,
                ceil_mode=True,
            )
            fixed_scale = F.avg_pool3d(
                fixed.float(),
                kernel_size=scale,
                stride=scale,
                ceil_mode=True,
            )
            mask_scale = (F.avg_pool3d(
                mask.float(),
                kernel_size=scale,
                stride=scale,
                ceil_mode=True,
            ).clamp(0.0, 1.0) if mask is not None else None)
        losses.append(
            _masked_global_ncc_loss(moving_scale, fixed_scale, mask_scale))
    return torch.stack(losses).mean()


def smoothness_loss(displacement: torch.Tensor) -> torch.Tensor:
    """Compute first-order displacement smoothness loss."""
    depth_gradient = displacement[:, :,
                                  1:, :, :] - displacement[:, :, :-1, :, :]
    height_gradient = displacement[:, :, :,
                                   1:, :] - displacement[:, :, :, :-1, :]
    width_gradient = displacement[:, :, :, :,
                                  1:] - displacement[:, :, :, :, :-1]

    return (depth_gradient.pow(2).mean() + height_gradient.pow(2).mean() +
            width_gradient.pow(2).mean())


def soft_dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute one soft Dice loss, retaining the historical reduction order.

    Args:
        prediction: Soft mask with values in ``[0, 1]``.
        target: Reference mask with the same shape.
        eps: Numerical stabilizer of the Dice ratio.

    Returns:
        One minus the soft Dice of the two masks.
    """
    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def label_group_mask(label: torch.Tensor,
                     class_ids: tuple[int, ...]) -> torch.Tensor:
    """Build a floating mask for a tuple of anatomical class IDs."""
    mask = torch.zeros_like(label, dtype=torch.bool)
    for class_id in class_ids:
        mask |= label == class_id
    return mask.float()


def _lncc_local_sums(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    window_size: int,
    *,
    square_method: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Compute shared LNCC convolution sums without changing square syntax."""
    channels = moving.shape[1]
    padding = window_size // 2
    kernel = torch.ones(
        (channels, 1, window_size, window_size, window_size),
        dtype=moving.dtype,
        device=moving.device,
    )
    moving_sum = F.conv3d(moving, kernel, padding=padding, groups=channels)
    fixed_sum = F.conv3d(fixed, kernel, padding=padding, groups=channels)
    moving_squared = moving.square() if square_method else moving * moving
    fixed_squared = fixed.square() if square_method else fixed * fixed
    moving_sq_sum = F.conv3d(moving_squared,
                             kernel,
                             padding=padding,
                             groups=channels)
    fixed_sq_sum = F.conv3d(fixed_squared,
                            kernel,
                            padding=padding,
                            groups=channels)
    cross_sum = F.conv3d(moving * fixed,
                         kernel,
                         padding=padding,
                         groups=channels)
    return moving_sum, fixed_sum, moving_sq_sum, fixed_sq_sum, cross_sum


def lncc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    window_size: int = 9,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute local normalized cross-correlation loss for 3D images.

    Args:
        moving: Warped moving image of shape ``[B, C, D, H, W]``.
        fixed: Fixed image with the same shape.
        window_size: Edge length of the cubic correlation window.
        eps: Numerical stabilizer of the variance product.

    Returns:
        Negative mean squared local correlation, in ``[-1, 0]``.

    Raises:
        ValueError: If the images differ in shape or are not five-dimensional.
    """
    if moving.shape != fixed.shape:
        raise ValueError(f"moving and fixed must have the same shape, got "
                         f"{moving.shape} and {fixed.shape}.")

    if moving.ndim != 5:
        raise ValueError(
            f"Expected tensors with shape [B, C, D, H, W], got {moving.shape}.")

    window_volume = float(window_size**3)
    moving_sum, fixed_sum, moving_sq_sum, fixed_sq_sum, cross_sum = (
        _lncc_local_sums(moving, fixed, window_size, square_method=False))

    moving_mean = moving_sum / window_volume
    fixed_mean = fixed_sum / window_volume

    cross = (cross_sum - fixed_mean * moving_sum - moving_mean * fixed_sum +
             moving_mean * fixed_mean * window_volume)
    moving_var = (moving_sq_sum - 2.0 * moving_mean * moving_sum +
                  moving_mean * moving_mean * window_volume)
    fixed_var = (fixed_sq_sum - 2.0 * fixed_mean * fixed_sum +
                 fixed_mean * fixed_mean * window_volume)

    lncc = (cross * cross) / (moving_var * fixed_var + eps)

    return -lncc.mean()


def masked_lncc_loss(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 9,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute LNCC while excluding windows outside the valid overlap.

    Args:
        moving: Warped moving image of shape ``[B, C, D, H, W]``.
        fixed: Fixed image with the same shape.
        mask: Validity mask of shape ``[B, 1, D, H, W]``.
        window_size: Edge length of the cubic correlation window.
        eps: Numerical stabilizer of the variance product.

    Returns:
        Negative squared local correlation averaged with the fraction of valid
        voxels in each window as weight.

    Raises:
        ValueError: If the images or the mask have inconsistent shapes.
    """
    if moving.shape != fixed.shape:
        raise ValueError("moving and fixed must have identical shapes.")
    if mask.ndim != 5 or mask.shape[0] != moving.shape[0]:
        raise ValueError("mask must have shape [B,1,D,H,W].")

    channels = moving.shape[1]
    padding = window_size // 2
    window_volume = float(window_size**3)
    moving_sum, fixed_sum, moving_sq_sum, fixed_sq_sum, cross_sum = (
        _lncc_local_sums(moving, fixed, window_size, square_method=True))
    moving_mean = moving_sum / window_volume
    fixed_mean = fixed_sum / window_volume
    cross = (cross_sum - fixed_mean * moving_sum - moving_mean * fixed_sum +
             moving_mean * fixed_mean * window_volume)
    moving_var = (moving_sq_sum - 2.0 * moving_mean * moving_sum +
                  moving_mean.square() * window_volume)
    fixed_var = (fixed_sq_sum - 2.0 * fixed_mean * fixed_sum +
                 fixed_mean.square() * window_volume)
    lncc = cross.square() / (moving_var * fixed_var + eps)

    local_validity = F.avg_pool3d(
        mask.float(),
        kernel_size=window_size,
        stride=1,
        padding=padding,
    )
    local_validity = local_validity.expand(-1, channels, -1, -1, -1)
    return -(lncc * local_validity).sum() / local_validity.sum().clamp_min(eps)
