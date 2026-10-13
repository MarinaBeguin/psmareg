"""Small synthetic volumes shared by the unit tests."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from psmareg.config.fine_tuning_config import (
    FineTuningConfig,
    load_fine_tuning_config,
)
from psmareg.config.synthetic_pretraining_config import (
    SyntheticPretrainingConfig,
    load_synthetic_pretraining_config,
)
from psmareg.models.common.registration.cascades import (
    AffineCascadeSettings,
    AffineTransMorphV2CascadeRegistration,
)
from psmareg.training.fine_tuning_batch import EpochWeights

SHAPE = (32, 32, 32)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def labeled_scan(shift: int = 0, seed: int = 0) -> dict[str, torch.Tensor]:
    """Build one CT/PET scan with anatomy and lesion labels.

    Args:
        shift: Offset of every structure along the last axis, in voxels.
        seed: Seed of the image texture.

    Returns:
        ``ct``, ``pet``, ``ct_label``, and ``pet_label`` tensors of shape
        ``[1, 1, 32, 32, 32]``; images lie in ``[0, 1]``.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (1, 1, *SHAPE)
    ct_label = torch.zeros(shape, dtype=torch.long)
    ct_label[..., 4:28, 5:27, 4 + shift:24 + shift] = 1
    ct_label[..., 8:14, 9:15, 7 + shift:13 + shift] = 2
    ct_label[..., 16:24, 14:22, 12 + shift:20 + shift] = 5
    pet_label = torch.zeros(shape, dtype=torch.long)
    pet_label[..., 9:13, 10:14, 8 + shift:12 + shift] = 1
    pet_label[..., 18:21, 16:19, 14 + shift:17 + shift] = 1
    body = (ct_label > 0).float()
    ct = (0.2 + 0.5 * torch.rand(shape, generator=generator)) * body
    ct = (ct + 0.1 * ct_label).clamp(0.0, 1.0)
    pet = 0.2 * torch.rand(shape, generator=generator) * body
    pet = (pet + 0.7 * pet_label).clamp(0.0, 1.0)
    return {"ct": ct, "pet": pet, "ct_label": ct_label, "pet_label": pet_label}


def paired_batch(shift: int = 3) -> dict[str, torch.Tensor | list[str]]:
    """Build a fixed/moving batch whose moving scan is shifted along x."""
    fixed = labeled_scan(0, seed=1)
    moving = labeled_scan(shift, seed=2)
    batch: dict[str, torch.Tensor | list[str]] = {"subject": ["subject_a"]}
    for prefix, scan in (("fixed", fixed), ("moving", moving)):
        for name, tensor in scan.items():
            batch[f"{prefix}_{name}"] = tensor
    return batch


def save_nifti(array: np.ndarray, path: Path) -> Path:
    """Write an array as a NIfTI file with an identity affine."""
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
    return path


def small_fine_tuning_config(**overrides: object) -> FineTuningConfig:
    """Load the final recipe with sizes adapted to the 32-voxel test pair."""
    values: dict[str, object] = {
        "num_ct_classes": 6,
        "scaling_steps": 4,
        "deterministic_translation_search_radius": 4,
        "mind_loss_downsample": 2,
        "macro_ct_label_downsample": 2,
        "smoothness_downsample": 2,
        "pet_lesion_occupancy_downsample": 1,
        "pretrained_checkpoint": None,
        "compatible_pretrained_checkpoint": None,
        "paired_cache_dir": None,
    }
    values.update(overrides)
    return load_fine_tuning_config(
        REPOSITORY_ROOT / "psmareg" / "config" / "fine_tuning_final.py", values)


def small_final_model(config: FineTuningConfig,
                      seed: int = 0) -> AffineTransMorphV2CascadeRegistration:
    """Build the final cascade with the bounds of ``config``."""
    torch.manual_seed(seed)
    return AffineTransMorphV2CascadeRegistration(
        AffineCascadeSettings(
            max_velocity=config.architecture.max_velocity,
            max_residual_disp=config.architecture.max_residual_disp,
            scaling_steps=config.architecture.scaling_steps))


def final_epoch_weights(config: FineTuningConfig) -> EpochWeights:
    """Return the loss weights reached at the end of every ramp."""
    return EpochWeights(
        lambda_macro_ct_labels=config.anatomical_loss.lambda_macro_ct_labels,
        lambda_pet_hot_region=config.pet_loss.lambda_pet_hot_region,
        lambda_pet_lesion_occupancy=(
            config.lesion_loss.lambda_pet_lesion_occupancy),
        lambda_pet_lesion_translation=(
            config.lesion_loss.lambda_pet_lesion_translation),
        lambda_pet_lesion_strain=config.lesion_loss.lambda_pet_lesion_strain)


def small_pretraining_config(stage: str = "c",
                             **overrides: object) -> SyntheticPretrainingConfig:
    """Load a stage recipe with deformations sized for the test scans."""
    values: dict[str, object] = {
        "scaling_steps": 4,
        "amp": False,
        "epochs": 4,
        "curriculum_total_epochs": 4,
        "max_translation": 3.0,
        "max_elastic_displacement": 2.0,
        "pose_max_translation": 2.0,
        "pose_min_region_fraction": 0.001,
        "target_generation_attempts": 4,
        "target_min_jacobian": 0.05,
        "metric_sampling_stride": 2,
        "jacobian_loss_stride": 2,
        "cache_dir": None,
        "exclude_subjects_manifest": None,
        "start_from": None,
    }
    values.update(overrides)
    return load_synthetic_pretraining_config(
        REPOSITORY_ROOT / "psmareg" / "config" /
        f"synthetic_pretraining_stage_{stage}.py", values)


def scan_batch(seed: int = 0) -> dict[str, torch.Tensor]:
    """Build one single-scan pretraining batch with labels."""
    return labeled_scan(seed=seed)
