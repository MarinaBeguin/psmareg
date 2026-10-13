"""Affine transformation utilities and heads for PSMAReg."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from psmareg.models.common.registration.blocks import (
    ConvBlock3D,
    conv_norm_activation,
    pairwise_ct_pet_features,
)
from psmareg.models.common.registration.geometry import (
    euler_rotation_matrix_xyz,
    voxel_grid_xyz,
)


def affine_parameters_to_displacement(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Convert per-sample affine parameters to a dense pull-back field.

    Translation is expressed in voxels, rotation in radians (x, y, z), and
    scale is multiplicative along each axis. The transform is centred in the
    image, matching the synthetic affine generator.

    Args:
        translation: Translations of shape ``[B, 3]`` in x/y/z voxels.
        rotation: Euler angles of shape ``[B, 3]`` in radians.
        scale: Scale factors of shape ``[B, 3]``.
        shape: Spatial shape ``(D, H, W)`` of the field.

    Returns:
        Field of shape ``[B, 3, D, H, W]`` whose rotation and scaling are
        centered on the volume.
    """
    batch_size = translation.shape[0]
    depth, height, width = shape
    dtype = translation.dtype
    device = translation.device

    coordinates = voxel_grid_xyz(shape, device=device, dtype=dtype).unsqueeze(0)
    center = torch.tensor(
        [(width - 1.0) / 2.0, (height - 1.0) / 2.0, (depth - 1.0) / 2.0],
        dtype=dtype,
        device=device,
    ).view(1, 3, 1, 1, 1)
    centered = (coordinates - center).expand(batch_size, -1, -1, -1, -1)

    matrix = euler_rotation_matrix_xyz(rotation) @ torch.diag_embed(scale)
    transformed = torch.einsum(
        "bij,bjdhw->bidhw",
        matrix,
        centered,
    )
    return transformed - centered + translation[:, :, None, None, None]


def body_center_translation(
    fixed_ct: torch.Tensor,
    moving_ct: torch.Tensor,
    threshold: float = 0.02,
    max_translation: float = 64.0,
) -> torch.Tensor:
    """Estimate the pull-back translation from robust CT body centroids.

    Args:
        fixed_ct: Fixed CT of shape ``[B, C, D, H, W]``.
        moving_ct: Moving CT with the same shape.
        threshold: Normalized intensity above which a voxel is body.
        max_translation: Bound of each translation component, in voxels.

    Returns:
        Translation of shape ``[B, 3]`` from the fixed to the moving body
        center.

    Raises:
        ValueError: If the two volumes differ in shape or are not five-
            dimensional.
    """
    if fixed_ct.shape != moving_ct.shape or fixed_ct.ndim != 5:
        raise ValueError("fixed_ct and moving_ct must share shape [B,C,D,H,W].")
    depth, height, width = fixed_ct.shape[-3:]
    coordinates = voxel_grid_xyz((depth, height, width),
                                 device=fixed_ct.device,
                                 dtype=fixed_ct.dtype).unsqueeze(0)

    def centroid(image: torch.Tensor) -> torch.Tensor:
        """Return the voxel centroid of the thresholded body mask."""
        mask = (image[:, :1] > threshold).to(image.dtype)
        mass = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        return (mask * coordinates).sum(dim=(2, 3, 4)) / mass

    # Pull-back convention: output x samples moving(x + displacement).
    translation = centroid(moving_ct) - centroid(fixed_ct)
    return translation.clamp(-max_translation, max_translation)


class AffineHeadBase(nn.Module):
    """Shared bounds, identity initialization, and parameter decoding."""

    parameters_head: nn.Sequential

    def __init__(self, max_translation: float, max_rotation_degrees: float,
                 max_scaling: float) -> None:
        super().__init__()
        self.max_translation = max_translation
        self.max_rotation_radians = math.radians(max_rotation_degrees)
        self.max_scaling = max_scaling

    def forward(
        self,
        x: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict an affine displacement; implemented by each head.

        Raises:
            NotImplementedError: Always; concrete heads override this method.
        """
        raise NotImplementedError

    def reset_to_identity(self) -> None:
        """Zero the last layer so that the head predicts the identity.

        Raises:
            TypeError: If the parameter head does not end with a linear layer.
        """
        final_layer = self.parameters_head[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError(
                "The affine parameter head must end with nn.Linear.")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def _decode_affine(
        self, raw: torch.Tensor,
        output_shape: tuple[int, int,
                            int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Map raw head outputs to a bounded affine field and parameters."""
        bounded = torch.tanh(raw)
        translation = bounded[:, :3] * self.max_translation
        rotation = bounded[:, 3:6] * self.max_rotation_radians
        scale = 1.0 + bounded[:, 6:9] * self.max_scaling
        displacement = affine_parameters_to_displacement(
            translation=translation,
            rotation=rotation,
            scale=scale,
            shape=output_shape)
        parameters = torch.cat((translation, rotation, scale), dim=1)
        return displacement, parameters


class AffineRegistrationHead(AffineHeadBase):
    """Predict a global affine transform for each fixed/moving pair."""

    def __init__(
        self,
        in_channels: int = 4,
        max_translation: float = 30.0,
        max_rotation_degrees: float = 15.0,
        max_scaling: float = 0.10,
    ) -> None:
        super().__init__(max_translation, max_rotation_degrees, max_scaling)
        self.encoder = nn.Sequential(
            *conv_norm_activation(in_channels,
                                  16,
                                  kernel_size=5,
                                  stride=2,
                                  padding=2),
            *conv_norm_activation(16, 32, stride=2),
            *conv_norm_activation(32, 64, stride=2),
        )
        self.parameters_head = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 9),
        )
        self.reset_to_identity()

    def forward(
        self,
        x: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a bounded affine displacement from global image features.

        Args:
            x: CT/PET pair of shape ``[B, 4, D, H, W]``.
            output_shape: Requested ``(D, H, W)`` displacement size.

        Returns:
            Affine displacement and decoded affine parameters.
        """
        # Bounding the input size keeps this global stage inexpensive.
        low = F.interpolate(
            x,
            size=(32, 32, 32),
            mode="trilinear",
            align_corners=False,
        )
        raw = self.parameters_head(self.encoder(low).flatten(1))
        return self._decode_affine(raw, output_shape)


class StrongAffineRegistrationHead(AffineHeadBase):
    """Stronger global affine head with pairwise CT/PET feature engineering."""

    def __init__(
        self,
        max_translation: float = 30.0,
        max_rotation_degrees: float = 15.0,
        max_scaling: float = 0.10,
    ) -> None:
        super().__init__(max_translation, max_rotation_degrees, max_scaling)
        self.input_fusion = nn.Sequential(
            *conv_norm_activation(10, 24),
            *conv_norm_activation(24, 24),
        )
        self.encoder = nn.Sequential(
            *conv_norm_activation(24, 32, kernel_size=5, stride=2, padding=2),
            *conv_norm_activation(32, 48, stride=2),
            *conv_norm_activation(48, 64, stride=2),
            *conv_norm_activation(64, 96, stride=2),
            ConvBlock3D(96, 128),
        )
        self.parameters_head = nn.Sequential(
            nn.Linear(256, 192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(192, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(96, 9),
        )
        self.reset_to_identity()

    @staticmethod
    def _pairwise_features(x: torch.Tensor) -> torch.Tensor:
        """Build the CT/PET pairwise input features."""
        return pairwise_ct_pet_features(x)

    def forward(
        self,
        x: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a bounded affine displacement from pairwise features.

        Args:
            x: CT/PET pair of shape ``[B, 4, D, H, W]``.
            output_shape: Requested ``(D, H, W)`` displacement size.

        Returns:
            Affine displacement and decoded affine parameters.
        """
        low = F.interpolate(
            self._pairwise_features(x),
            size=(48, 48, 48),
            mode="trilinear",
            align_corners=False,
        )
        features = self.encoder(self.input_fusion(low))
        pooled_avg = F.adaptive_avg_pool3d(features, output_size=1).flatten(1)
        pooled_max = F.adaptive_max_pool3d(features, output_size=1).flatten(1)
        raw = self.parameters_head(torch.cat((pooled_avg, pooled_max), dim=1))
        return self._decode_affine(raw, output_shape)
