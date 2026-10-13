"""Preprocessing utilities for PSMAReg CT/PET volumes.
Same preprocessing as Learn2Reg26_PSMA was applied to the dataset"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

CT_WINDOW = (-300.0, 300.0)
PET_UPPER_QUANTILE = 0.99


def ct_window_normalize(
        volume: np.ndarray,
        window: tuple[float, float] = (-300.0, 300.0),
) -> np.ndarray:
    """Clip CT values to a window and normalize to [0, 1].

    Raises:
        ValueError: If the upper bound of the window is not above the lower one.
    """
    lower, upper = window

    if upper <= lower:
        raise ValueError(f"Invalid CT window: {window}")

    volume_float = np.asarray(volume, dtype=np.float32)
    normalized = (np.clip(volume_float, lower, upper) - lower) / (upper - lower)
    normalized[~np.isfinite(volume_float)] = 0.0

    return normalized.astype(np.float32)


def get_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary mask."""
    labeled_mask, number_of_components = ndimage.label(mask)

    if number_of_components == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = ndimage.sum(
        mask,
        labeled_mask,
        index=np.arange(1, number_of_components + 1),
    )
    largest_component_label = int(np.argmax(component_sizes)) + 1

    return labeled_mask == largest_component_label


def slice_center_mask(shape_xy: tuple[int, int],
                      fraction: float = 0.5) -> np.ndarray:
    """Create a central 2D mask used to identify the body component."""
    height, width = shape_xy

    y_margin = int((1.0 - fraction) * height / 2.0)
    x_margin = int((1.0 - fraction) * width / 2.0)

    mask = np.zeros((height, width), dtype=bool)
    mask[y_margin:height - y_margin, x_margin:width - x_margin] = True

    return mask


def select_body_components(
    slice_mask: np.ndarray,
    center_mask: np.ndarray,
    previous_support: np.ndarray | None,
) -> np.ndarray:
    """Select body-like connected components on one axial slice.

    Args:
        slice_mask: Binary body candidates of one axial slice.
        center_mask: Central region that the body is expected to cover.
        previous_support: Body mask of the neighboring slice, if already
            tracked.

    Returns:
        Components overlapping the center or the previous slice.
    """
    labeled_slice, number_of_components = ndimage.label(slice_mask)

    if number_of_components == 0:
        return np.zeros_like(slice_mask, dtype=bool)

    selected = np.zeros_like(slice_mask, dtype=bool)

    for component_label in range(1, number_of_components + 1):
        component = labeled_slice == component_label

        overlaps_center = np.logical_and(component, center_mask).any()
        overlaps_previous = (previous_support is not None and
                             np.logical_and(component, previous_support).any())

        if overlaps_center or overlaps_previous:
            selected |= component

    return selected


def remove_bed(volume: np.ndarray) -> np.ndarray:
    """Suppress scanner bed/table while preserving the patient body."""
    image = np.asarray(volume, dtype=np.float32).copy()

    body_candidate = image >= -700.0
    body_candidate = ndimage.binary_opening(
        body_candidate,
        structure=np.ones((3, 3, 3), dtype=bool),
    )

    tracked_mask = np.zeros_like(body_candidate, dtype=bool)
    center_mask = slice_center_mask(body_candidate.shape[:2])
    middle_slice = body_candidate.shape[2] // 2

    previous_support: np.ndarray | None = None
    for z_index in range(middle_slice, -1, -1):
        current = select_body_components(
            slice_mask=body_candidate[:, :, z_index],
            center_mask=center_mask,
            previous_support=previous_support,
        )
        tracked_mask[:, :, z_index] = current
        if current.any():
            previous_support = current

    previous_support = None
    for z_index in range(middle_slice + 1, body_candidate.shape[2]):
        current = select_body_components(
            slice_mask=body_candidate[:, :, z_index],
            center_mask=center_mask,
            previous_support=previous_support,
        )
        tracked_mask[:, :, z_index] = current
        if current.any():
            previous_support = current

    if tracked_mask.sum() == 0:
        tracked_mask = get_largest_connected_component(body_candidate)
    else:
        tracked_mask = ndimage.binary_fill_holes(tracked_mask)
        tracked_mask = get_largest_connected_component(tracked_mask)

    tracked_mask = ndimage.binary_closing(
        tracked_mask,
        structure=np.ones((5, 5, 3), dtype=bool),
    )
    tracked_mask = ndimage.binary_fill_holes(tracked_mask)
    tracked_mask = ndimage.binary_dilation(
        tracked_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
    )

    background_value = float(np.percentile(image, 0.5))
    image[tracked_mask == 0] = background_value

    return image.astype(np.float32)


def preprocess_ct(volume: np.ndarray) -> np.ndarray:
    """Apply bed suppression and CT window normalization."""
    bed_removed = remove_bed(volume)
    return ct_window_normalize(bed_removed, window=CT_WINDOW)


def preprocess_pet(
    volume: torch.Tensor,
    upper_quantile: float = PET_UPPER_QUANTILE,
) -> torch.Tensor:
    """Normalize one PET scan robustly to [0, 1].

    Each scan is scaled independently by the requested quantile of its
    strictly positive finite voxels. Negative and non-finite values map to 0.

    Raises:
        ValueError: If ``upper_quantile`` is outside ``(0, 1]``.
    """
    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0, 1].")

    finite_volume = torch.nan_to_num(
        volume.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    positive_values = finite_volume[finite_volume > 0.0]
    if positive_values.numel() == 0:
        return torch.zeros_like(finite_volume)

    scale = torch.quantile(
        positive_values,
        upper_quantile,
    ).clamp_min(1e-6)
    return torch.clamp(finite_volume / scale, min=0.0, max=1.0)
