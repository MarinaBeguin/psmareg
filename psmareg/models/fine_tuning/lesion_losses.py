"""PET lesion occupancy, strain, and similarity-masking objectives."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label as connected_component_label

from psmareg.config.fine_tuning_config import LesionLossConfig
from psmareg.models.common.registration.geometry import voxel_grid_xyz
from psmareg.models.common.registration.displacement import (
    resize_voxel_displacement,)
from psmareg.models.common.registration.warping import warp_image_3d


def _coordinate_grid_voxel(depth: int, height: int, width: int, *,
                           device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
    """Create a voxel-coordinate grid matching a displacement tensor."""
    return voxel_grid_xyz((depth, height, width), device=device,
                          dtype=dtype).unsqueeze(0)


def _soft_lattice_occupancy(source_coordinates: torch.Tensor) -> torch.Tensor:
    """Softly count how often each moving voxel is sampled by a pull-back field.

    ``source_coordinates`` has shape ``(1, 3, D, H, W)`` in voxel units and
    stores, for each fixed voxel, the moving-space coordinate sampled by
    ``grid_sample``.  The returned tensor approximates the nearest-neighbor
    occupancy count ``n(y)=#{x: x+u(x)=y}``, using trilinear deposition so the
    loss remains useful for backpropagation.
    """
    if source_coordinates.shape[0] != 1 or source_coordinates.shape[1] != 3:
        raise ValueError(
            "_soft_lattice_occupancy expects shape (1, 3, D, H, W).")
    _, _, depth, height, width = source_coordinates.shape
    coords = source_coordinates[0]
    x = coords[0].reshape(-1)
    y = coords[1].reshape(-1)
    z = coords[2].reshape(-1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    z0 = torch.floor(z)
    fx = x - x0
    fy = y - y0
    fz = z - z0
    x0 = x0.long()
    y0 = y0.long()
    z0 = z0.long()
    occupancy = torch.zeros(depth * height * width,
                            device=source_coordinates.device,
                            dtype=source_coordinates.dtype)
    for dz in (0, 1):
        wz = fz if dz else 1.0 - fz
        zi = z0 + dz
        valid_z = (zi >= 0) & (zi < depth)
        for dy in (0, 1):
            wy = fy if dy else 1.0 - fy
            yi = y0 + dy
            valid_y = (yi >= 0) & (yi < height)
            for dx in (0, 1):
                wx = fx if dx else 1.0 - fx
                xi = x0 + dx
                valid = valid_z & valid_y & (xi >= 0) & (xi < width)
                if not bool(valid.any().detach().cpu()):
                    continue
                linear_index = zi[valid] * (
                    height * width) + yi[valid] * width + xi[valid]
                weight = (wx * wy * wz)[valid]
                occupancy.scatter_add_(0, linear_index, weight)
    return occupancy.reshape(1, 1, depth, height, width)


def _downsample_lesion_inputs(
    moving_pet: torch.Tensor,
    moving_pet_label: torch.Tensor,
    displacement: torch.Tensor,
    valid_mask: torch.Tensor | None,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Downsample the lesion-loss inputs onto the occupancy grid.

    Args:
        moving_pet: Moving PET of shape ``[B, 1, D, H, W]``.
        moving_pet_label: Moving lesion labels of shape ``[B, D, H, W]``.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        valid_mask: Optional validity mask of shape ``[B, 1, D, H, W]``.
        factor: Downsampling factor along each axis.

    Returns:
        PET, labels, displacement in low-resolution voxels, and validity mask.
    """
    if factor <= 1:
        return moving_pet.float(), moving_pet_label, displacement, valid_mask
    moving_pet_low = F.avg_pool3d(moving_pet.float(),
                                  kernel_size=factor,
                                  stride=factor,
                                  ceil_mode=True)
    target_size = moving_pet_low.shape[-3:]
    moving_label_low = F.interpolate(moving_pet_label.unsqueeze(1).float(),
                                     size=target_size,
                                     mode="nearest").squeeze(1).long()
    displacement_low = resize_voxel_displacement(displacement,
                                                 target_size,
                                                 align_corners=False)
    valid_mask_low = F.interpolate(
        valid_mask.float(), size=target_size,
        mode="nearest") if valid_mask is not None else None
    return moving_pet_low, moving_label_low, displacement_low, valid_mask_low


def _largest_lesion_components(
        lesion: np.ndarray, min_voxels: int,
        max_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Label 6-connected lesion components and keep the largest ones.

    Args:
        lesion: Binary lesion mask of shape ``[D, H, W]``.
        min_voxels: Smallest retained component size.
        max_components: Largest number of retained components.

    Returns:
        Component label map and retained labels sorted by decreasing size.
    """
    component_map, num_components = connected_component_label(lesion)
    if num_components <= 0:
        return component_map, np.empty(0, dtype=np.int64)
    component_sizes = np.bincount(component_map.reshape(-1))
    component_ids = np.flatnonzero(component_sizes >= min_voxels)
    component_ids = component_ids[component_ids > 0]
    component_ids = component_ids[np.argsort(
        component_sizes[component_ids])[::-1][:max_components]]
    return component_map, component_ids


def _lesion_component_weight(component: np.ndarray,
                             config: LesionLossConfig) -> float:
    """Compute the size- and border-dependent weight of one component.

    Args:
        component: Binary component mask of shape ``[D, H, W]``.
        config: Lesion-preservation settings.

    Returns:
        Weight applied to the occupancy and translation penalties.
    """
    weight = 1.0
    if config.pet_lesion_adaptive_size_weighting:
        voxels = float(component.sum())
        small, large = (config.pet_lesion_small_voxels,
                        config.pet_lesion_large_voxels)
        if voxels <= small:
            weight = config.pet_lesion_small_weight
        elif voxels >= large:
            weight = config.pet_lesion_large_weight
        else:
            fraction = (voxels - small) / (large - small)
            weight = config.pet_lesion_small_weight + fraction * (
                config.pet_lesion_large_weight - config.pet_lesion_small_weight)
    if config.pet_lesion_edge_guard_margin_voxels > 0:
        coordinates = np.argwhere(component)
        far_side = np.asarray(component.shape) - 1 - coordinates.max(axis=0)
        border_distance = min(int(np.min(coordinates)), int(np.min(far_side)))
        if border_distance <= config.pet_lesion_edge_guard_margin_voxels:
            weight = max(weight, config.pet_lesion_edge_guard_weight)
    return weight


def _lesion_halo(component: np.ndarray, valid: np.ndarray | None,
                 iterations: int) -> np.ndarray:
    """Return the 6-connected peri-lesional halo of one component.

    Args:
        component: Binary component mask of shape ``[D, H, W]``.
        valid: Optional binary validity mask with the same shape.
        iterations: Number of binary-dilation iterations.

    Returns:
        Binary mask of valid non-lesion voxels surrounding the component.
    """
    halo = binary_dilation(component, iterations=iterations) & ~component
    return halo & valid if valid is not None else halo


def _lesion_component_penalties(
    occupancy: torch.Tensor,
    displacement: torch.Tensor,
    pet: torch.Tensor,
    component_mask: torch.Tensor,
    halo_mask: torch.Tensor | None,
    halo_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute occupancy and translation penalties of one lesion component.

    Args:
        occupancy: Soft sampling occupancy of shape ``[1, 1, D, H, W]``.
        displacement: Pull-back field of shape ``[1, 3, D, H, W]``.
        pet: Non-negative moving PET of shape ``[1, 1, D, H, W]``.
        component_mask: Binary component mask of shape ``[1, 1, D, H, W]``.
        halo_mask: Optional binary halo mask with the same shape.
        halo_weight: Coefficient of the halo penalties.

    Returns:
        Occupancy penalty and translation-consistency penalty.
    """
    mask_sum = component_mask.sum()
    occ_error = (occupancy - 1.0).square()
    occ_loss = (occ_error * component_mask).sum() / mask_sum.clamp_min(1.0)
    pet_weight = pet * component_mask
    pet_weight_sum = pet_weight.sum()
    if float(pet_weight_sum.detach().cpu()) >= 1e-06:
        pet_occ_loss = ((occupancy - 1.0).abs() *
                        pet_weight).sum() / pet_weight_sum.clamp_min(1e-06)
        occ_loss = 0.5 * occ_loss + 0.5 * pet_occ_loss
    mean_disp = (displacement * component_mask).sum(
        dim=(2, 3, 4), keepdim=True) / mask_sum.clamp_min(1.0)
    trans_loss = ((displacement - mean_disp).square() *
                  component_mask).sum() / (3.0 * mask_sum.clamp_min(1.0))
    if halo_mask is not None:
        halo_sum = halo_mask.sum().clamp_min(1.0)
        halo_weight_tensor = torch.as_tensor(halo_weight,
                                             device=displacement.device,
                                             dtype=displacement.dtype)
        halo_occ_loss = (occ_error * halo_mask).sum() / halo_sum
        halo_trans_loss = ((displacement - mean_disp).square() *
                           halo_mask).sum() / (3.0 * halo_sum)
        occ_loss = occ_loss + halo_weight_tensor * halo_occ_loss
        trans_loss = trans_loss + halo_weight_tensor * halo_trans_loss
    return occ_loss, trans_loss


def pet_lesion_occupancy_and_translation_losses(
    moving_pet: torch.Tensor,
    moving_pet_label: torch.Tensor,
    displacement: torch.Tensor,
    valid_mask: torch.Tensor | None,
    config: LesionLossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Protect PET lesions from nearest-neighbor MTV/TLG aliasing.

    The official scorer warps moving PET labels and PET intensities with
    nearest interpolation and compares MTV/TLG before and after. Scalar MTV/TLG
    losses are weak because many local deformations keep the global sums
    similar. This loss instead asks every moving lesion voxel to be sampled
    approximately once by the pull-back deformation, and a companion
    translation loss keeps each connected lesion component moving close to
    rigidly.

    Args:
        moving_pet: Moving PET of shape ``[B, 1, D, H, W]``.
        moving_pet_label: Moving lesion labels of shape ``[B, D, H, W]``.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        valid_mask: Optional validity mask of shape ``[B, 1, D, H, W]``.
        config: Lesion-preservation settings.

    Returns:
        Occupancy loss, translation-consistency loss, and the number of lesion
        components that contributed.
    """
    (moving_pet_low, moving_label_low, displacement_low,
     valid_mask_low) = _downsample_lesion_inputs(
         moving_pet, moving_pet_label, displacement, valid_mask,
         config.pet_lesion_occupancy_downsample)
    grid_shape = displacement_low.shape[-3:]
    identity = _coordinate_grid_voxel(*grid_shape,
                                      device=displacement_low.device,
                                      dtype=displacement_low.dtype)

    def to_mask(array: np.ndarray) -> torch.Tensor:
        """Convert a NumPy component mask to a ``[1, 1, D, H, W]`` tensor."""
        return torch.from_numpy(array.astype(np.float32)).to(
            device=displacement_low.device,
            dtype=displacement_low.dtype).view(1, 1, *grid_shape)

    occupancy_losses: list[torch.Tensor] = []
    translation_losses: list[torch.Tensor] = []
    for sample_idx in range(displacement_low.shape[0]):
        lesion_np = (moving_label_low[sample_idx] > 0).detach().cpu().numpy()
        valid_np = None
        if valid_mask_low is not None:
            valid_np = valid_mask_low[sample_idx,
                                      0].detach().cpu().numpy() > 0.5
            lesion_np = lesion_np & valid_np
        component_map, component_ids = _largest_lesion_components(
            lesion_np, config.pet_lesion_occupancy_min_voxels,
            config.pet_lesion_occupancy_max_components)
        if component_ids.size == 0:
            continue
        occupancy = _soft_lattice_occupancy(
            identity + displacement_low[sample_idx:sample_idx + 1])
        disp_sample = displacement_low[sample_idx:sample_idx + 1]
        pet_sample = moving_pet_low[sample_idx:sample_idx + 1].detach().abs()
        for component_id in component_ids:
            component_np = component_map == int(component_id)
            halo_mask = None
            if (config.pet_lesion_halo_voxels > 0 and
                    config.pet_lesion_halo_weight > 0.0):
                halo_np = _lesion_halo(component_np, valid_np,
                                       config.pet_lesion_halo_voxels)
                if bool(halo_np.any()):
                    halo_mask = to_mask(halo_np)
            occ_loss, trans_loss = _lesion_component_penalties(
                occupancy, disp_sample, pet_sample, to_mask(component_np),
                halo_mask, config.pet_lesion_halo_weight)
            component_weight = torch.as_tensor(_lesion_component_weight(
                component_np, config),
                                               device=displacement_low.device,
                                               dtype=displacement_low.dtype)
            occupancy_losses.append(component_weight * occ_loss)
            translation_losses.append(component_weight * trans_loss)
    if occupancy_losses:
        loss_occupancy = torch.stack(occupancy_losses).mean()
        loss_translation = torch.stack(translation_losses).mean()
    else:
        loss_occupancy = torch.zeros((), device=displacement.device)
        loss_translation = torch.zeros((), device=displacement.device)
    count_tensor = torch.as_tensor(float(len(occupancy_losses)),
                                   device=displacement.device,
                                   dtype=displacement.dtype)
    return (loss_occupancy, loss_translation, count_tensor.detach())


def pet_lesion_strain_loss(moving_pet_label: torch.Tensor,
                           displacement: torch.Tensor,
                           valid_mask: torch.Tensor | None, *,
                           downsample_factor: int,
                           min_voxels: int) -> torch.Tensor:
    """Penalize non-rigid local deformation inside PET-label lesions.

    The hot-region Jacobian term mostly controls expansion/compression. A lesion
    can still be sheared while keeping a near-unit determinant, which is bad for
    nearest-neighbor MTV/TLG. This term computes a low-resolution deformation
    gradient F = I + grad(u) inside moving PET labels and penalizes F^T F - I.
    That encourages small lesions to travel approximately as rigid blobs while
    allowing surrounding tissue to absorb the deformation.

    Args:
        moving_pet_label: Moving lesion labels of shape ``[B, D, H, W]``.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        valid_mask: Optional valid sampling mask.
        downsample_factor: Spatial reduction for the strain calculation.
        min_voxels: Minimum lesion support required to apply the loss.

    Returns:
        Scalar strain penalty, or zero when lesion support is insufficient.
    """
    if downsample_factor > 1:
        target_size = tuple((max(1, math.ceil(size / downsample_factor))
                             for size in displacement.shape[-3:]))
        disp = resize_voxel_displacement(displacement,
                                         target_size,
                                         align_corners=False)
        lesion = F.interpolate(moving_pet_label.unsqueeze(1).float(),
                               size=target_size,
                               mode="nearest")
        valid = F.interpolate(
            valid_mask.float(), size=target_size,
            mode="nearest") if valid_mask is not None else None
    else:
        disp = displacement
        lesion = moving_pet_label.unsqueeze(1).float()
        valid = valid_mask
    lesion = (lesion > 0).float()
    if valid is not None:
        lesion = lesion * (valid > 0.5).float()
    if min(disp.shape[-3:]) < 2 or float(
            lesion.detach().sum().cpu()) < float(min_voxels):
        return torch.zeros((), device=displacement.device)
    weight = lesion[:, :, :-1, :-1, :-1]
    if float(weight.detach().sum().cpu()) < float(min_voxels):
        return torch.zeros((), device=displacement.device)
    du_dx = disp[:, :, :-1, :-1, 1:] - disp[:, :, :-1, :-1, :-1]
    du_dy = disp[:, :, :-1, 1:, :-1] - disp[:, :, :-1, :-1, :-1]
    du_dz = disp[:, :, 1:, :-1, :-1] - disp[:, :, :-1, :-1, :-1]
    f_xx = 1.0 + du_dx[:, 0:1]
    f_xy = du_dy[:, 0:1]
    f_xz = du_dz[:, 0:1]
    f_yx = du_dx[:, 1:2]
    f_yy = 1.0 + du_dy[:, 1:2]
    f_yz = du_dz[:, 1:2]
    f_zx = du_dx[:, 2:3]
    f_zy = du_dy[:, 2:3]
    f_zz = 1.0 + du_dz[:, 2:3]
    c_xx = f_xx.square() + f_yx.square() + f_zx.square() - 1.0
    c_yy = f_xy.square() + f_yy.square() + f_zy.square() - 1.0
    c_zz = f_xz.square() + f_yz.square() + f_zz.square() - 1.0
    c_xy = f_xx * f_xy + f_yx * f_yy + f_zx * f_zy
    c_xz = f_xx * f_xz + f_yx * f_yz + f_zx * f_zz
    c_yz = f_xy * f_xz + f_yy * f_yz + f_zy * f_zz
    frob = c_xx.square() + c_yy.square() + c_zz.square() + 2.0 * (
        c_xy.square() + c_xz.square() + c_yz.square())
    return (frob * weight).sum() / weight.sum().clamp_min(1.0)


def mask_pet_similarity_around_moving_lesions(
        base_mask: torch.Tensor | None, moving_pet_label: torch.Tensor,
        displacement: torch.Tensor, *, dilate: int, downsample: int,
        padding_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove warped moving lesions from a PET similarity mask.

    Args:
        base_mask: Optional original PET similarity mask.
        moving_pet_label: Moving lesion labels of shape ``[B, D, H, W]``.
        displacement: Pull-back field of shape ``[B, 3, D, H, W]``.
        dilate: Dilation radius around warped lesions, in voxels.
        downsample: Spatial reduction for mask construction.
        padding_mode: Padding used when warping lesion masks.

    Returns:
        Retained-similarity mask and detached removed-voxel fraction.
    """
    lesion_mask = (moving_pet_label > 0).float().unsqueeze(1)
    output_shape = lesion_mask.shape[-3:]
    if downsample > 1:
        low_shape = tuple((max(1, size // downsample) for size in output_shape))
        lesion_mask = F.interpolate(lesion_mask, size=low_shape, mode="nearest")
        displacement = resize_voxel_displacement(displacement.detach().float(),
                                                 low_shape,
                                                 align_corners=False)
        dilate = int(math.ceil(float(dilate) / float(downsample)))
    else:
        displacement = displacement.detach()
    warped_lesion = warp_image_3d(lesion_mask,
                                  displacement,
                                  interpolation_mode="bilinear",
                                  padding_mode=padding_mode).clamp(0.0, 1.0)
    if dilate > 0:
        warped_lesion = F.max_pool3d(warped_lesion,
                                     kernel_size=2 * dilate + 1,
                                     stride=1,
                                     padding=dilate)
    keep_mask = (1.0 - warped_lesion).clamp(0.0, 1.0)
    if downsample > 1:
        keep_mask = F.interpolate(keep_mask,
                                  size=output_shape,
                                  mode="trilinear",
                                  align_corners=False).clamp(0.0, 1.0)
        warped_lesion = F.interpolate(warped_lesion,
                                      size=output_shape,
                                      mode="trilinear",
                                      align_corners=False).clamp(0.0, 1.0)
    if base_mask is not None:
        keep_mask = keep_mask * base_mask.float().clamp(0.0, 1.0)
    return (keep_mask, warped_lesion.mean().detach())
