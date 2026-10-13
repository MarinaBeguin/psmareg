/bin/bash: line 1: config: command not found
/bin/bash: line 1: +m.group(1).upper()+: command not found
# Training parameter reference

This document describes every uppercase constant accepted in `fine_tuning_final.py`. The schema converts these constants into an immutable `FineTuningConfig`. The **final value** column is read from `psmareg/config/fine_tuning_final.py`; the **library default** is the fallback defined by the corresponding dataclass in `psmareg/config/fine_tuning_config.py`. For a new experiment, copy `fine_tuning_final.py` and change only the copy.

## How weights and switches interact

- Every `lambda_*` is a loss weight. Setting it to `0` disables that term while leaving its secondary parameters inert.
- A Boolean master switch such as `train_augmentation`, `adaptive_pet_loss`, or `deterministic_translation_prealign` must be enabled before the associated settings have an effect.
- Downsampling factors are spatial factors: larger values reduce memory and computation but also reduce spatial precision.
- Ramp durations progressively introduce a loss during the stated number of epochs; `0` applies its full weight immediately.
- `resume_from` restores a real-pair training state and therefore cannot be combined with the synthetic initialization checkpoints.
- Paths are interpreted relative to the repository root when the launcher starts from that directory.

## Data and sampling

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `train_manifest` | `manifests/paired/train_pairs.csv` | same path | filesystem path | CSV listing the paired examinations used for training. Each row must follow the contract expected by PSMARegPairDataset. |
| `val_manifest` | `manifests/paired/val_pairs.csv` | same path | filesystem path | CSV listing validation pairs. These subjects are not used for optimizer updates and drive checkpoint selection. |
| `output_dir` | `results/fine_tuning` | same path | filesystem path | Directory receiving checkpoints, metrics.csv and run artifacts. Use a new directory for every independent experiment. |
| `paired_cache_dir` | `./psmareg_paired_cache` | `None` | path or None | Optional cache for preprocessed paired volumes; override with `PSMAREG_PAIRED_CACHE_DIR` or set to None to disable. |
| `paired_cache_float16` | `True` | `False` | true or false | Store normalized CT and PET images in the pair cache as float16 to reduce disk space and I/O. Labels keep their discrete representation. |
| `patient_balanced_sampling` | `True` | `False` | true or false | Sample patients uniformly instead of sampling pair rows uniformly, preventing patients with many pairs from dominating training. |

## Initialization and resume

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `pretrained_checkpoint` | `results/pretraining/stage_c/best_val.pt` | `None` | path or None | Primary synthetic-registration checkpoint loaded before real-pair training. This is normally the Stage C global-expert checkpoint. |
| `compatible_pretrained_checkpoint` | `results/pretraining/stage_b/best_val.pt` | `None` | path or None | Second checkpoint loaded with shape-compatible, non-strict matching; used to transfer selected Stage B weights. |
| `compatible_pretrained_prefixes` | `'refiner.'` | `''` | Comma-separated prefixes | Comma-separated state-dict prefixes allowed from the compatible checkpoint. Empty means all compatible tensors. |
| `resume_from` | `None` | `None` | Path or None; cannot coexist with pretraining checkpoints | Resume a real-pair run including model, optimizer and epoch. Mutually exclusive with both pretraining checkpoints. |

## Model architecture

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `num_ct_classes` | `118` | `118` | integer >= 1 | Number of detailed CT segmentation classes used by the CT Dice supervision. |
| `max_velocity` | `12.0` | `8.0` | > 0 voxels | Maximum magnitude, in voxels, of the predicted stationary velocity field before integration. The final checkpoint records 12.0; 8.0 is only the library default. |
| `max_residual_disp` | `4.0` | `4.0` | > 0 voxels | Maximum magnitude, in voxels, of the residual displacement predicted by the refinement stage. |
| `scaling_steps` | `7` | `7` | integer >= 1 | Number of scaling-and-squaring steps used to integrate the stationary velocity field. More steps improve integration accuracy but cost time. |

## Optimization

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `epochs` | `250` | `100` | integer >= 1 | Maximum number of complete passes over the training dataset. |
| `batch_size` | `1` | `1` | integer >= 1 | Number of image pairs processed together. Full-resolution 3D training is memory intensive; the final experiment uses one. |
| `lr` | `1e-05` | `0.0001` | > 0 | Initial AdamW learning rate applied to affine and deformable parameter groups. |
| `gradient_clip_norm` | `1.0` | `0.0` | >= 0; 0 disables | Maximum global gradient norm applied after backpropagation and before the AdamW step. The final checkpoint records 1.0. |
| `patience` | `30` | `20` | integer >= 0 | Number of validation epochs without improvement allowed before early stopping. |
| `lr_scheduler_patience` | `6` | `0` | integer >= 0; 0 disables | Validation plateaus tolerated before reducing the learning rate. Zero disables the scheduler. |
| `lr_scheduler_factor` | `0.5` | `0.5` | 0 < value < 1 | Multiplicative learning-rate reduction applied after a plateau. |
| `min_lr` | `1e-07` | `1e-07` | > 0 | Lower bound imposed by the learning-rate scheduler. |

## Training augmentation

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `train_augmentation` | `True` | `False` | true or false | Master switch for training-time spatial and intensity augmentation. Augmentation is never applied during validation. |
| `train_spatial_aug_probability` | `0.35` | `0.0` | 0 to 1 | Probability that a training pair receives the random affine spatial augmentation. |
| `train_aug_max_translation` | `3.0` | `3.0` | >= 0 voxels | Maximum absolute random translation per axis, in voxels. |
| `train_aug_max_rotation_degrees` | `2.0` | `2.0` | >= 0 degrees | Maximum absolute random rotation around each axis, in degrees. |
| `train_aug_max_scaling` | `0.02` | `0.02` | >= 0 relative scale | Maximum relative isotropic/axis scaling deviation; 0.02 corresponds to approximately ±2%. |
| `train_intensity_aug_probability` | `0.35` | `0.0` | 0 to 1 | Probability of applying independent CT/PET intensity perturbations to a training pair. |

## Image similarity losses

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `lambda_pet` | `0.004` | `0.2` | >= 0; 0 disables this term | Weight of the direct PET intensity similarity loss. Keep small because uptake can genuinely change longitudinally. |
| `lambda_mind_ct` | `0.024` | `0.0` | >= 0; 0 disables this term | Weight of the CT MIND-SSC descriptor alignment loss. |
| `lambda_mind_pet` | `0.001` | `0.0` | >= 0; 0 disables this term | Weight of the PET MIND-SSC descriptor alignment loss. Keep smaller than CT because lesions and physiological uptake may change. |
| `mind_loss_downsample` | `4` | `2` | integer >= 1 | Spatial downsampling factor used before computing MIND losses. Larger values reduce memory and spatial precision. |
| `mind_radius` | `1` | `1` | integer >= 0 | Patch radius used by the MIND descriptor. |
| `mind_dilation` | `2` | `2` | integer >= 1 | Voxel spacing between MIND descriptor comparisons. |
| `lambda_multiscale_corr_ct` | `0.016` | `0.0` | >= 0; 0 disables this term | Weight of global multiscale CT correlation. |
| `lambda_multiscale_corr_pet` | `0.002` | `0.0` | >= 0; 0 disables this term | Weight of global multiscale PET correlation. |
| `multiscale_corr_scales` | `(2, 4, 8)` | `(2, 4, 8)` | tuple of positive integers | Downsampling scales at which global correlation is evaluated. |

## Anatomical supervision

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `lambda_ct_dice` | `0.85` | `0.5` | >= 0; 0 disables this term | Weight of detailed multi-class CT-label Dice supervision. |
| `ct_dice_loss_downsample` | `1` | `1` | integer >= 1 | Downsampling factor for detailed CT Dice. One means full resolution. |
| `lambda_macro_ct_labels` | `0.03` | `0.0` | >= 0; 0 disables this term | Weight of weak Dice supervision over broad, stable CT anatomical groups. |
| `macro_ct_label_downsample` | `4` | `2` | integer >= 1 | Downsampling factor used for macro CT-label supervision. |
| `macro_ct_label_preset` | `'stable_no_bladder'` | `'stable_no_bladder'` | stable, stable_no_bladder, skeleton_only, organs_only | Selection of anatomical groups included in macro supervision. stable_no_bladder excludes the mobile bladder. |
| `macro_ct_label_ramp_epochs` | `20` | `20` | integer >= 0 | Epochs over which the macro CT-label weight rises linearly from zero to its configured value. |

## PET protection and adaptive similarity

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `lambda_pet_hot_region` | `0.001` | `0.0` | >= 0; 0 disables this term | Weight of the PET similarity term focused on robust high-uptake regions. |
| `pet_hot_quantile` | `0.97` | `0.98` | 0 < value <= 1 | Lower per-volume PET quantile defining the hot-region soft mask. |
| `pet_hot_upper_quantile` | `0.995` | `1.0` | 0 < value <= 1 | Upper PET quantile cap for the hot-region mask, limiting domination by extreme bladder uptake. |
| `pet_hot_warped_weight` | `0.04` | `0.25` | >= 0 | Contribution of warped-moving hot voxels in addition to fixed hot voxels. |
| `pet_hot_smooth_kernel` | `7` | `5` | positive odd integer | Odd 3D average-pooling kernel applied before hot-region detection. |
| `pet_hot_ramp_epochs` | `5` | `0` | integer >= 0 | Epochs over which the hot-region loss weight rises from zero. |
| `pet_loss_upper_quantile` | `0.995` | `1.0` | 0 < value <= 1 | Upper PET quantile included in direct PET similarity; values below one exclude extreme uptake. |
| `pet_loss_quantile_downsample` | `1` | `1` | integer >= 1 | Downsampling used only to estimate the PET quantile mask; the final similarity remains on its normal grid. |
| `pet_mask_similarity_near_lesions` | `True` | `False` | true or false | Remove direct PET/MIND/correlation supervision around warped moving lesions so changing lesions are not forced to match. |
| `pet_lesion_similarity_mask_dilate` | `6` | `5` | integer >= 0 | Dilation radius, in voxels on the mask grid, around lesions excluded from PET similarity. |
| `pet_lesion_similarity_mask_downsample` | `1` | `1` | integer >= 1 | Downsampling used to construct the lesion exclusion mask. |
| `adaptive_pet_loss` | `True` | `False` | true or false | Scale PET-based losses according to correspondence between fixed and warped-moving hot regions. |
| `adaptive_pet_hot_quantile` | `0.97` | `0.97` | 0 < value <= 1 | PET quantile used to build hot masks for the non-gradient adaptive gate. |
| `adaptive_pet_smooth_kernel` | `7` | `7` | positive odd integer | Odd smoothing kernel used before adaptive hot-region detection. |
| `adaptive_pet_downsample` | `1` | `1` | integer >= 1 | Downsampling used for the non-gradient adaptive correspondence calculation. |
| `adaptive_pet_min_weight` | `0.06` | `0.15` | 0 to 1 | Minimum multiplier retained for PET losses when hot-region correspondence is poor. |
| `adaptive_pet_low_dice` | `0.04` | `0.05` | must be below high threshold | Hot-region Dice at or below which the minimum adaptive PET weight is used. |
| `adaptive_pet_high_dice` | `0.3` | `0.35` | must exceed low threshold | Hot-region Dice at or above which the full PET weight is restored. |

## Lesion-aware losses

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `lambda_pet_lesion_occupancy` | `0.18` | `0.0` | >= 0; 0 disables this term | Weight of lattice-occupancy protection against skipped or multiply sampled lesion voxels. |
| `lambda_pet_lesion_translation` | `0.065` | `0.0` | >= 0; 0 disables this term | Weight encouraging each lesion component to move approximately as a coherent block. |
| `lambda_pet_lesion_strain` | `0.05` | `0.0` | >= 0; 0 disables this term | Weight penalizing local shear and shape distortion inside PET lesions. |
| `pet_lesion_occupancy_downsample` | `2` | `4` | integer >= 1 | Downsampling factor for occupancy, translation and strain calculations. |
| `pet_lesion_occupancy_min_voxels` | `4` | `3` | integer >= 1 | Minimum lesion-component size on the downsampled grid. Smaller components are ignored. |
| `pet_lesion_occupancy_max_components` | `12` | `12` | integer >= 1 | Maximum number of largest lesion components evaluated per sample. |
| `pet_lesion_occupancy_ramp_epochs` | `6` | `10` | integer >= 0 | Epochs over which lesion-protection loss weights rise from zero. |
| `pet_lesion_adaptive_size_weighting` | `True` | `False` | true or false | Give stronger protection to small lesions and softer protection to very large hot components. |
| `pet_lesion_small_voxels` | `80.0` | `80.0` | > 0 and below large boundary | Downsampled component-size boundary below which the small-lesion weight applies. |
| `pet_lesion_large_voxels` | `1500.0` | `1500.0` | greater than small boundary | Downsampled component-size boundary above which the large-component weight applies. |
| `pet_lesion_small_weight` | `2.2` | `1.8` | > 0 | Relative lesion-loss multiplier for small components. |
| `pet_lesion_large_weight` | `0.35` | `0.45` | > 0 | Relative lesion-loss multiplier for very large components. |
| `pet_lesion_halo_voxels` | `2` | `0` | integer >= 0 | Dilation radius around each lesion component for a weaker transition penalty. Zero disables the halo. |
| `pet_lesion_halo_weight` | `0.25` | `0.25` | >= 0 | Relative occupancy/translation weight assigned to the lesion halo. |
| `pet_lesion_edge_guard_margin_voxels` | `4` | `0` | integer >= 0 | Border distance defining lesions at risk of leaving the field of view. Zero disables the edge guard. |
| `pet_lesion_edge_guard_weight` | `1.6` | `1.0` | > 0 | Minimum component weight assigned to lesions identified as field-of-view risks. |

## Regularization

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `lambda_smooth` | `0.02` | `0.01` | >= 0; 0 disables this term | Weight of displacement-field spatial smoothness regularization. |
| `smoothness_downsample` | `4` | `1` | integer >= 1 | Downsampling factor before computing smoothness regularization. |
| `lambda_overlap` | `2.0` | `0.0` | >= 0; 0 disables this term | Weight of the hinge penalty when valid fixed/moving overlap becomes too small. |
| `minimum_valid_overlap` | `0.9` | `0.9` | 0 to 1 | Target minimum fraction of valid overlap. The overlap penalty activates below this value. |

## Deterministic pre-alignment

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `deterministic_translation_prealign` | `True` | `False` | true or false | Enable deterministic coarse translation search before the learned cascade. |
| `deterministic_translation_downsample` | `4` | `4` | integer >= 1 | Downsampling factor used during coarse translation search. |
| `deterministic_translation_search_radius` | `12` | `8` | integer >= 0 | Maximum search offset along each axis on the search grid. |
| `deterministic_translation_search_step` | `2` | `2` | integer >= 1 | Step between candidate translations on the search grid. |
| `deterministic_translation_max` | `120.0` | `80.0` | >= 0 voxels | Maximum accepted translation magnitude in full-resolution voxels. |
| `deterministic_translation_feature_mode` | `'ct'` | `'ct'` | ct or mind | Feature used to score candidate translations: raw normalized CT or CT MIND. |
| `deterministic_translation_padding_mode` | `'border'` | `'auto'` | auto, border or zeros | Padding used while evaluating translation candidates. auto follows the model mode. |
| `deterministic_translation_input_padding_mode` | `'border'` | `'auto'` | auto, border or zeros | Padding used to prewarp inputs passed to the learned model. |

## Runtime and logging

| Parameter | Final value | Library default | Allowed values / unit | Description |
|---|---:|---:|---|---|
| `num_workers` | `2` | `4` | integer >= 0 | Number of DataLoader worker processes. Zero loads data in the training process. |
| `no_persistent_workers` | `False` | `False` | true or false | Restart DataLoader workers each epoch instead of keeping them alive. |
| `log_challenge_metrics` | `True` | `False` | true or false | Compute official-like challenge metrics during validation. This is informative but relatively expensive. |
| `seed` | `42` | `42` | integer | Seed passed to PyTorch and NumPy. Reproducibility still depends on deterministic CUDA operations and identical data ordering. |

## Recommended editing workflow

1. Copy `psmareg/config/fine_tuning_final.py` to a new, clearly named configuration.
2. Give the run a new `data.output_dir`; never reuse a directory containing another experiment.
3. Change one scientifically motivated group at a time.
4. Run a short validation job before launching the full training.
5. Keep the configuration file with the resulting checkpoints and metrics.
