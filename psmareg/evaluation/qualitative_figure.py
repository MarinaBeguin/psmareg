"""Panels and layout of the qualitative results figure.

The figure shows representative cases by four columns, with one CT row and
one PET row for each case:

    fixed | moving | fixed-moving difference | fixed-warped difference

It works best on the internal labeled validation split, where PET lesion
labels are available and can be overlaid as contours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib

# Backend selection must precede pyplot import on headless cluster nodes.
# pylint: disable=wrong-import-position
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# pylint: enable=wrong-import-position

# Cyan, used for the moving and warped PET lesions that drive MTV/TLG.
MOVING_LESION_COLOR = "#00d7ff"

DIFFERENCE_CMAP = "inferno"


def add_orientation_markers(ax: plt.Axes) -> None:
    """Label radiological left/right and anterior/posterior directions."""
    for label, x, y in (("R", 0.03, 0.5), ("L", 0.97, 0.5), ("A", 0.5, 0.97),
                        ("P", 0.5, 0.03)):
        ax.text(x,
                y,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.65,
                    "pad": 0.7
                })


def slice_crop(
    fixed_ct: np.ndarray,
    moving_ct: np.ndarray,
    warped_ct: np.ndarray,
    z: int,
    margin: int,
) -> tuple[slice, slice]:
    """Return a 2D body crop for a given axial slice.

    Args:
        fixed_ct: Fixed CT, slice axis first.
        moving_ct: Moving CT with the same shape.
        warped_ct: Warped CT with the same shape.
        z: Displayed axial slice.
        margin: Margin kept around the body, in voxels.

    Returns:
        Row and column slices, covering the whole image when no body is found.
    """
    slices = [fixed_ct[z], moving_ct[z], warped_ct[z]]
    masks = []
    for image in slices:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            continue
        threshold = float(np.percentile(finite, 18.0))
        masks.append(image > threshold)
    if not masks:
        return slice(None), slice(None)
    mask = np.any(np.stack(masks), axis=0)
    if int(mask.sum()) < 32:
        return slice(None), slice(None)
    yy, xx = np.where(mask)
    y0 = max(int(np.min(yy)) - margin, 0)
    y1 = min(int(np.max(yy)) + margin + 1, mask.shape[0])
    x0 = max(int(np.min(xx)) - margin, 0)
    x1 = min(int(np.max(xx)) + margin + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


@dataclass(frozen=True)
class SliceView:
    """Axial slice and in-plane crop shared by every panel of one case."""

    z: int
    crop: tuple[slice, slice]

    def of(self, volume: np.ndarray) -> np.ndarray:
        """Return the displayed part of ``volume``."""
        ys, xs = self.crop
        return volume[self.z][ys, xs]


def _hide_ticks(ax: plt.Axes) -> None:
    """Remove ticks and use a black background behind the image."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")


def draw_scan(
    ax: plt.Axes,
    image: np.ndarray,
    lesion: np.ndarray | None,
    view: SliceView,
    window: tuple[float, float],
    cmap: str,
    contour_color: str = "#00e5ff",
) -> None:
    """Draw one anatomical or functional slice and optional lesion contour.

    Args:
        ax: Target axes.
        image: Volume to display.
        lesion: Optional lesion label volume drawn as a contour.
        view: Displayed slice and crop.
        window: Lower and upper display intensities.
        cmap: Matplotlib colormap name.
        contour_color: Color of the lesion contour.
    """
    ax.imshow(view.of(image),
              cmap=cmap,
              vmin=window[0],
              vmax=window[1],
              origin="upper")
    if lesion is not None and np.nanmax(view.of(lesion)) > 0:
        lesion_view = view.of(lesion) > 0.5
        ax.contour(lesion_view, levels=[0.5], colors=["black"], linewidths=1.8)
        ax.contour(lesion_view,
                   levels=[0.5],
                   colors=[contour_color],
                   linewidths=1.0)
    _hide_ticks(ax)


def draw_contours(
    ax: plt.Axes,
    lesions: list[tuple[np.ndarray | None, str]],
    view: SliceView,
    linewidth: float = 1.0,
) -> None:
    """Overlay one or more PET lesion contours on an existing panel.

    Args:
        ax: Target axes.
        lesions: Lesion volumes and their contour colors.
        view: Displayed slice and crop.
        linewidth: Width of the colored contour.
    """
    for lesion, color in lesions:
        if lesion is None:
            continue
        lesion_view = view.of(lesion)
        if np.nanmax(lesion_view) > 0:
            mask = lesion_view > 0.5
            ax.contour(mask,
                       levels=[0.5],
                       colors=["black"],
                       linewidths=linewidth + 0.8)
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=linewidth)


def draw_difference(
    ax: plt.Axes,
    fixed: np.ndarray,
    moving_or_warped: np.ndarray,
    view: SliceView,
    vmax: float,
    lesion_contours: list[tuple[np.ndarray | None, str]] | None = None,
    display_mask: np.ndarray | None = None,
) -> Any:
    """Draw the absolute difference between fixed and moving/warped slices.

    Args:
        ax: Target axes.
        fixed: Fixed volume.
        moving_or_warped: Volume compared with the fixed one.
        view: Displayed slice and crop.
        vmax: Upper bound of the difference color scale.
        lesion_contours: Optional lesion volumes and their contour colors.
        display_mask: Optional mask outside which the difference is hidden.

    Returns:
        Matplotlib image, usable as a colorbar mappable.
    """
    difference = np.abs(view.of(fixed) - view.of(moving_or_warped))
    if display_mask is not None:
        difference = np.where(display_mask, difference, 0.0)
    image = ax.imshow(difference,
                      cmap=DIFFERENCE_CMAP,
                      vmin=0.0,
                      vmax=max(float(vmax), 1e-6),
                      origin="upper")
    if lesion_contours:
        draw_contours(ax, lesion_contours, view)
    _hide_ticks(ax)
    return image


def pull_image_axes_together(axes: np.ndarray, ct_row: int,
                             pet_row: int) -> None:
    """Anchor paired CT/PET panels toward each other.

    Matplotlib preserves image aspect ratio inside each subplot.  With medical
    slices this can leave vertical blank space that is not affected much by
    GridSpec hspace.  Anchoring CT panels to the bottom and PET panels to the
    top makes each CT/PET pair visually compact without stretching the scans.

    Args:
        axes: Grid of figure axes.
        ct_row: Row of the CT panels.
        pet_row: Row of the PET panels.
    """
    for column in range(4):
        axes[ct_row, column].set_anchor("S")
        axes[pet_row, column].set_anchor("N")


def body_difference_mask(
    fixed_ct: np.ndarray,
    moving_or_warped_ct: np.ndarray,
    view: SliceView,
    mode: str = "fixed",
) -> np.ndarray:
    """Mask difference maps to the anatomical support on the displayed slice.

    Args:
        fixed_ct: Fixed CT, slice axis first.
        moving_or_warped_ct: CT compared with the fixed one.
        view: Displayed slice and crop.
        mode: ``fixed``, ``union``, or ``intersection`` of the two bodies.

    Returns:
        Boolean mask on the displayed crop.
    """
    fixed_mask = view.of(fixed_ct) > 0.035
    other_mask = view.of(moving_or_warped_ct) > 0.035
    if mode == "union":
        return fixed_mask | other_mask
    if mode == "intersection":
        return fixed_mask & other_mask
    return fixed_mask


def manually_set_ct_pet_gap(
    fig: plt.Figure,
    axes: np.ndarray,
    case_count: int,
    gap: float = 0.004,
) -> None:
    """Force a visible but small gap between CT and PET rows.

    ``imshow(aspect='equal')`` can ignore GridSpec spacing visually because the
    active image box is smaller than the allocated subplot slot.  This routine
    runs after Matplotlib has computed the active axes boxes, then explicitly
    moves PET panels close to their paired CT panels.

    Args:
        fig: Figure holding the axes.
        axes: Grid of figure axes.
        case_count: Number of cases in the figure.
        gap: Vertical gap in figure coordinates.
    """
    fig.canvas.draw()
    for row_index in range(case_count):
        ct_row = 3 * row_index
        pet_row = ct_row + 1
        ct_positions = [axes[ct_row, col].get_position() for col in range(4)]
        pet_positions = [axes[pet_row, col].get_position() for col in range(4)]
        ct_bottom = min(pos.y0 for pos in ct_positions)
        pet_top = max(pos.y1 for pos in pet_positions)
        shift = (ct_bottom - gap) - pet_top
        for col in range(4):
            pos = axes[pet_row, col].get_position()
            axes[pet_row, col].set_position(
                [pos.x0, pos.y0 + shift, pos.width, pos.height])


def manually_compact_case_blocks(
    fig: plt.Figure,
    axes: np.ndarray,
    case_count: int,
    gap: float = 0.020,
) -> None:
    """Move whole patient blocks upward to enforce a compact inter-case gap.

    Args:
        fig: Figure holding the axes.
        axes: Grid of figure axes.
        case_count: Number of cases in the figure.
        gap: Vertical gap between two cases in figure coordinates.
    """
    fig.canvas.draw()
    for row_index in range(1, case_count):
        previous_metric_row = 3 * (row_index - 1) + 2
        current_ct_row = 3 * row_index
        previous_bottom = min(axes[previous_metric_row, col].get_position().y0
                              for col in range(4))
        current_top = max(
            axes[current_ct_row, col].get_position().y1 for col in range(4))
        shift = (previous_bottom - gap) - current_top
        for local_row in range(3):
            absolute_row = 3 * row_index + local_row
            for col in range(5):
                pos = axes[absolute_row, col].get_position()
                axes[absolute_row, col].set_position(
                    [pos.x0, pos.y0 + shift, pos.width, pos.height])


def blank_axis(ax: plt.Axes) -> None:
    """Hide an unused subplot without changing the layout grid."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_facecolor("white")


def metric_caption(
    metrics: dict[tuple[str, str], dict[str, float]],
    row: Mapping[str, str],
    panel: str,
    case_label: str,
) -> str:
    """Format a panel label with optional challenge scores.

    Args:
        metrics: Challenge scores by subject and follow-up.
        row: Manifest row of the case.
        panel: Panel letter of the case.
        case_label: Display name of the case.

    Returns:
        Caption, limited to the panel label when no score is known.
    """
    prefix = f"({panel}) {case_label}"
    key = (row["subject"], row["followup_index"])
    if key not in metrics:
        return prefix
    m = metrics[key]
    return (f"({panel}) {case_label}: "
            f"DSC {m['DSC']:.3f} | "
            f"HD95 {m['HD95']:.2f} mm | "
            f"MTV {m['MTV']:.2f}% | "
            f"TLG {m['TLG']:.2f}%")
