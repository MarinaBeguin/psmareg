"""Synthetic deformation generation for pretraining stages A, B, and C."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from psmareg.models.common.registration.geometry import (
    euler_rotation_matrix_xyz,
    voxel_grid_xyz,
)
from psmareg.models.common.registration.jacobian import jacobian_determinant
from psmareg.models.common.registration.warping import (
    compose_displacements,
    integrate_stationary_velocity,
)


@dataclass(frozen=True)
class CurriculumSettings:
    """Difficulty distribution for synthetic deformation sampling."""

    min_scale: float = 0.25
    warmup_fraction: float = 0.2
    ramp_fraction: float = 0.5
    easy_probability: float = 0.2
    extra_hard_probability: float = 0.0
    extra_hard_max_scale: float = 1.0


def curriculum_deformation_scale(
    epoch: int,
    total_epochs: int,
    device: torch.device,
    settings: CurriculumSettings,
) -> float:
    """Sample deformation severity from an expanding curriculum.

    Args:
        epoch: Current epoch index.
        total_epochs: Number of epochs over which the curriculum expands.
        device: Device of the random draws.
        settings: Difficulty distribution of the curriculum.

    Returns:
        Multiplier applied to the synthetic deformation magnitudes.

    Raises:
        ValueError: If a curriculum setting is out of range.
    """
    min_scale = settings.min_scale
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if not 0.0 < min_scale <= 1.0:
        raise ValueError("curriculum_min_scale must be in (0, 1].")
    if (settings.warmup_fraction < 0.0 or settings.ramp_fraction <= 0.0 or
            settings.warmup_fraction + settings.ramp_fraction > 1.0):
        raise ValueError(
            "Curriculum fractions must be non-negative and sum to at most 1.")
    if settings.easy_probability < 0.0 or settings.extra_hard_probability < 0.0:
        raise ValueError("Curriculum probabilities must be non-negative.")
    hard_threshold = (settings.easy_probability +
                      settings.extra_hard_probability)
    if hard_threshold > 1.0:
        raise ValueError("Curriculum probabilities must sum to at most 1.")
    if settings.extra_hard_max_scale < 1.0:
        raise ValueError("extra_hard_max_scale must be at least 1.")

    progress = max(0.0, min(float(epoch) / float(total_epochs), 1.0))
    if progress < settings.warmup_fraction:
        maximum_scale = min_scale
    elif progress < settings.warmup_fraction + settings.ramp_fraction:
        ramp_progress = ((progress - settings.warmup_fraction) /
                         settings.ramp_fraction)
        maximum_scale = min_scale + ramp_progress * (1.0 - min_scale)
    else:
        maximum_scale = 1.0

    random_value = float(torch.rand((), device=device).cpu())
    if maximum_scale < 1.0:
        return min_scale + random_value * (maximum_scale - min_scale)

    # Mature curriculum: retain some easy examples, focus most batches on
    # difficult transforms, and optionally expose a small extra-hard tail.
    interval_value = float(torch.rand((), device=device).cpu())
    if random_value < settings.easy_probability:
        return min_scale + interval_value * (0.75 - min_scale)
    if random_value < hard_threshold:
        return 1.0 + interval_value * (settings.extra_hard_max_scale - 1.0)
    return 0.75 + interval_value * 0.25


@dataclass(frozen=True)
class AffineSamplingSettings:
    """Bounds for a randomly sampled synthetic affine field."""

    max_translation: float
    max_rotation_degrees: float
    max_scaling: float


def random_affine_displacement(
    shape: tuple[int, int, int, int],
    settings: AffineSamplingSettings,
    device: torch.device,
) -> torch.Tensor:
    """Sample one bounded affine displacement per synthetic training item.

    Args:
        shape: Batch size and output depth, height, width in voxels.
        settings: Translation, rotation, and scaling bounds.
        device: Device for sampling and output.

    Returns:
        Displacement tensor of shape ``[B, 3, D, H, W]`` in voxel units.
    """
    batch_size, depth, height, width = shape
    max_translation = settings.max_translation
    max_rotation_degrees = settings.max_rotation_degrees
    max_scaling = settings.max_scaling
    coordinates = voxel_grid_xyz(
        (depth, height, width), device=device,
        dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

    center = torch.tensor(
        [(width - 1.0) / 2.0, (height - 1.0) / 2.0, (depth - 1.0) / 2.0],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1, 1)
    centered = coordinates - center

    displacement = torch.empty_like(coordinates)

    for batch_idx in range(batch_size):
        angles = (torch.empty(3, device=device).uniform_(-1.0, 1.0) *
                  math.radians(max_rotation_degrees))
        scale = 1.0 + torch.empty(3, device=device).uniform_(
            -max_scaling,
            max_scaling,
        )
        translation = torch.empty(3, device=device).uniform_(
            -max_translation,
            max_translation,
        )

        affine = euler_rotation_matrix_xyz(angles) @ torch.diag(scale)
        sample_centered = centered[batch_idx].view(3, -1)
        transformed = affine @ sample_centered
        transformed = transformed.view(3, depth, height, width)

        displacement[batch_idx] = transformed - centered[batch_idx]
        displacement[batch_idx] += translation.view(3, 1, 1, 1)

    return displacement


def _smooth_region_mask(mask: torch.Tensor,
                        downsample_factor: int) -> torch.Tensor:
    """Make a hard pseudo-region mask smooth enough for local-rigid blending."""
    if downsample_factor <= 1:
        return mask
    spatial_shape = mask.shape[-3:]
    low_shape = tuple(
        max(2, math.ceil(size / downsample_factor)) for size in spatial_shape)
    low = F.interpolate(
        mask,
        size=low_shape,
        mode="trilinear",
        align_corners=True,
    )
    return F.interpolate(
        low,
        size=spatial_shape,
        mode="trilinear",
        align_corners=True,
    ).clamp(0.0, 1.0)


def _random_local_affine_displacement(
    coordinates: torch.Tensor,
    center: torch.Tensor,
    max_translation: float,
    max_rotation_degrees: float,
    max_scaling: float,
) -> torch.Tensor:
    """Sample the dense field of a small affine transform about a center."""
    device = coordinates.device
    angles = (torch.empty(3, device=device).uniform_(-1.0, 1.0) *
              math.radians(max_rotation_degrees))
    scale = 1.0 + torch.empty(3, device=device).uniform_(
        -max_scaling, max_scaling)
    translation = torch.empty(3, device=device).uniform_(
        -max_translation,
        max_translation,
    )

    affine = euler_rotation_matrix_xyz(angles) @ torch.diag(scale)

    centered = coordinates - center.view(3, 1, 1, 1)
    transformed = affine @ centered.view(3, -1)
    transformed = transformed.view_as(centered)
    return transformed - centered + translation.view(3, 1, 1, 1)


@dataclass(frozen=True)
class PoseAwareSettings:
    """Bounds and smoothing of the articulated pseudo-rigid motion."""

    max_translation: float
    max_rotation_degrees: float
    max_scaling: float
    region_blur_downsample: int
    min_region_fraction: float


def _soft_body_quadrants(body: torch.Tensor,
                         coordinates: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split a body mask into four soft left/right, upper/lower regions."""
    _, depth, height, width = coordinates.shape
    x, z = coordinates[0], coordinates[2]
    body_mass = body.sum().clamp_min(1.0)
    center_x = (body.squeeze(0) * x).sum() / body_mass
    center_z = (body.squeeze(0) * z).sum() / body_mass
    x_softness = max(float(width) * 0.08, 1.0)
    z_softness = max(float(depth) * 0.10, 1.0)
    shape = (1, 1, depth, height, width)
    left = torch.sigmoid((center_x - x) / x_softness).view(shape)
    right = torch.sigmoid((x - center_x) / x_softness).view(shape)
    inferior = torch.sigmoid((z - center_z) / z_softness).view(shape)
    superior = torch.sigmoid((center_z - z) / z_softness).view(shape)
    return (
        body * left * inferior,
        body * right * inferior,
        body * left * superior,
        body * right * superior,
    )


def _blend_region_motions(region_masks: tuple[torch.Tensor,
                                              ...], coordinates: torch.Tensor,
                          settings: PoseAwareSettings) -> torch.Tensor:
    """Blend one random local affine motion per sufficiently large region."""
    _, depth, height, width = coordinates.shape
    device = coordinates.device
    weight_sum = torch.zeros(1, 1, depth, height, width, device=device)
    local_sum = torch.zeros(3, depth, height, width, device=device)
    minimum_voxels = settings.min_region_fraction * float(
        depth * height * width)
    for hard_region in region_masks:
        if float((hard_region
                  > 0.25).float().sum().detach().cpu()) < minimum_voxels:
            continue
        weight = _smooth_region_mask(
            hard_region,
            downsample_factor=settings.region_blur_downsample,
        ).squeeze(0)
        mass = weight.sum().clamp_min(1.0)
        center = torch.stack([
            (weight.squeeze(0) * coordinates[0]).sum() / mass,
            (weight.squeeze(0) * coordinates[1]).sum() / mass,
            (weight.squeeze(0) * coordinates[2]).sum() / mass,
        ])
        local_field = _random_local_affine_displacement(
            coordinates=coordinates,
            center=center,
            max_translation=settings.max_translation,
            max_rotation_degrees=settings.max_rotation_degrees,
            max_scaling=settings.max_scaling,
        )
        local_sum = local_sum + weight * local_field
        weight_sum = weight_sum + weight.unsqueeze(0)
    return local_sum / weight_sum.squeeze(0).clamp_min(1.0)


def random_pose_aware_displacement_from_labels(
    label: torch.Tensor,
    settings: PoseAwareSettings,
) -> torch.Tensor:
    """Generate smooth articulated pseudo-rigid motion for broad body regions.

    This is intentionally not a detailed biomechanical model. It creates a few
    large, soft, left/right and cranio-caudal body regions and applies different
    small local rigid/affine transforms to them. The goal is to teach the model
    that limbs/pelvis/shoulders can move differently from the trunk while
    keeping transitions smooth.

    Args:
        label: Anatomy labels of shape ``[B, 1, D, H, W]``.
        settings: Motion bounds and region smoothing.

    Returns:
        Displacement of shape ``[B, 3, D, H, W]``, zero for samples whose
        body is too small.

    Raises:
        ValueError: If ``label`` does not have five dimensions.
    """
    if label.ndim != 5:
        raise ValueError("Expected label shape [B, 1, D, H, W].")
    batch_size, _, depth, height, width = label.shape
    coordinates = voxel_grid_xyz((depth, height, width),
                                 device=label.device,
                                 dtype=torch.float32)
    displacement = torch.zeros(batch_size,
                               3,
                               depth,
                               height,
                               width,
                               device=label.device)
    for batch_idx in range(batch_size):
        body = (label[batch_idx:batch_idx + 1] > 0).float()
        if body.mean() < settings.min_region_fraction:
            continue
        displacement[batch_idx] = _blend_region_motions(
            _soft_body_quadrants(body, coordinates), coordinates, settings)
    return displacement


def random_elastic_velocity(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    max_displacement: float,
    control_point_spacing: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample a coarse random velocity and interpolate it to image resolution.

    Args:
        batch_size: Number of sampled fields.
        depth: Output depth in voxels.
        height: Output height in voxels.
        width: Output width in voxels.
        max_displacement: Bound applied to control-point velocities.
        control_point_spacing: Approximate spacing between control points.
        device: Device for sampling and output.

    Returns:
        Velocity tensor of shape ``[B, 3, D, H, W]`` in voxel units.
    """
    control_depth = max(2, math.ceil(depth / control_point_spacing) + 1)
    control_height = max(2, math.ceil(height / control_point_spacing) + 1)
    control_width = max(2, math.ceil(width / control_point_spacing) + 1)

    control_field = torch.randn(
        batch_size,
        3,
        control_depth,
        control_height,
        control_width,
        device=device,
    )
    control_field = control_field / control_field.abs().amax(
        dim=(1, 2, 3, 4),
        keepdim=True,
    ).clamp_min(1e-6)
    control_field = control_field * max_displacement

    return F.interpolate(
        control_field,
        size=(depth, height, width),
        mode="trilinear",
        align_corners=True,
    )


def random_synthetic_displacement_components(
    image: torch.Tensor,
    max_elastic_displacement: float,
    elastic_control_point_spacing: int,
    max_translation: float,
    max_rotation_degrees: float,
    max_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the composed synthetic field and its known affine component.

    Args:
        image: Image defining the batch size, spatial shape, and device.
        max_elastic_displacement: Largest elastic velocity component, in voxels.
        elastic_control_point_spacing: Spacing of the elastic control points.
        max_translation: Largest affine translation per axis, in voxels.
        max_rotation_degrees: Largest affine rotation per axis.
        max_scaling: Largest relative affine scale change.

    Returns:
        Composed affine-then-elastic field and its affine component, both of
        shape ``[B, 3, D, H, W]``.
    """
    batch_size, _, depth, height, width = image.shape
    affine_displacement = random_affine_displacement(
        shape=(batch_size, depth, height, width),
        settings=AffineSamplingSettings(max_translation, max_rotation_degrees,
                                        max_scaling),
        device=image.device,
    )
    elastic_velocity = random_elastic_velocity(
        batch_size=batch_size,
        depth=depth,
        height=height,
        width=width,
        max_displacement=max_elastic_displacement,
        control_point_spacing=elastic_control_point_spacing,
        device=image.device,
    )
    elastic_displacement = integrate_stationary_velocity(
        elastic_velocity,
        scaling_steps=7,
    )
    return (
        compose_displacements(
            first=affine_displacement,
            second=elastic_displacement,
        ),
        affine_displacement,
    )


def smooth_coarse_target(
    displacement: torch.Tensor,
    downsample_factor: int,
) -> torch.Tensor:
    """Remove local detail that the coarse stage cannot represent."""
    if downsample_factor <= 1:
        return displacement

    spatial_shape = displacement.shape[-3:]
    low_shape = tuple(
        max(2, math.ceil(size / downsample_factor)) for size in spatial_shape)
    low_resolution = F.interpolate(
        displacement,
        size=low_shape,
        mode="trilinear",
        align_corners=True,
    )
    return F.interpolate(
        low_resolution,
        size=spatial_shape,
        mode="trilinear",
        align_corners=True,
    )


@dataclass(frozen=True)
class SyntheticTargetSettings:
    """Affine, elastic, and Jacobian constraints for a synthetic target."""

    max_elastic_displacement: float
    elastic_control_point_spacing: int
    max_translation: float
    max_rotation_degrees: float
    max_scaling: float
    minimum_jacobian: float
    max_attempts: int
    jacobian_stride: int = 4


def safe_random_synthetic_displacement_components(
    image: torch.Tensor,
    settings: SyntheticTargetSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a safe composed target while retaining its affine component.

    Raises:
        ValueError: If ``max_attempts`` is below one.
        RuntimeError: If no sampled target reaches the minimum Jacobian.
    """
    max_attempts = settings.max_attempts
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    best: tuple[torch.Tensor, torch.Tensor] | None = None
    best_minimum = -float("inf")
    for _ in range(max_attempts):
        candidate, affine = random_synthetic_displacement_components(
            image=image,
            max_elastic_displacement=settings.max_elastic_displacement,
            elastic_control_point_spacing=settings.
            elastic_control_point_spacing,
            max_translation=settings.max_translation,
            max_rotation_degrees=settings.max_rotation_degrees,
            max_scaling=settings.max_scaling,
        )
        candidate_minimum = float(
            jacobian_determinant(
                candidate.detach().float(),
                sampling_stride=settings.jacobian_stride,
            ).min().cpu())
        if candidate_minimum > best_minimum:
            best = candidate, affine
            best_minimum = candidate_minimum
        if candidate_minimum >= settings.minimum_jacobian:
            return candidate, affine
    if best is None:
        raise RuntimeError("Synthetic target generation produced no candidate.")
    raise RuntimeError(
        "Could not generate a safe synthetic target after "
        f"{max_attempts} attempts (best minimum Jacobian={best_minimum:.6f}, "
        f"required={settings.minimum_jacobian:.6f}).")
