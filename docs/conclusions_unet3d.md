# PSMAReg – UNet3D Registration Experiments with CT Label Supervision

## Objective

Improve the Learn2Reg PSMAReg registration performance, with a primary focus on increasing the challenge DSC score while preserving PET-related metrics (MTV, TLG) and maintaining anatomically plausible deformations.

---

# Baseline: UNet3D + CT Dice Foreground

## Loss

```python
loss = (
    loss_ct
    + lambda_pet * loss_pet
    + lambda_smooth * loss_smooth
    + lambda_ct_dice * loss_ct_dice
)
```

where:

* `loss_ct`: LNCC loss on CT
* `loss_pet`: PET MSE
* `loss_smooth`: displacement smoothness penalty
* `loss_ct_dice`: foreground Dice on CT labels (`label > 0`)

Hyperparameters:

```text
lambda_pet      = 0.2
lambda_smooth   = 0.01
lambda_ct_dice  = 0.5
```

## Validation Results

| Metric        |  Value |
| ------------- | -----: |
| DSC           | 0.2793 |
| HD95          |  26.45 |
| MTV Error (%) |   4.74 |
| TLG Error (%) |   5.95 |
| NDV (%)       |   0.87 |

This remains the best-performing configuration tested so far.

---

# Per-Label CT Dice Analysis

## Motivation

The challenge DSC is computed independently for each anatomical label.

The original foreground Dice only optimizes:

```python
(label > 0)
```

which ignores the distinction between individual organs.

A per-label analysis was therefore performed on the training set.

## Dataset Statistics

Training labels contain:

```text
118 anatomical classes
labels 0 ... 117
```

Many labels are present in almost all volumes, while a few small structures appear less frequently.

## Observations

Large structures obtain relatively high Dice values:

```text
label 5   ≈ 0.69
label 11  ≈ 0.61
label 12  ≈ 0.67
label 80  ≈ 0.67
```

Small structures perform extremely poorly:

```text
label 23 ≈ 0.01
label 24 ≈ 0.01
label 90 ≈ 0.04
```

These labels often contain only a few tens of voxels.

## Conclusion

The overall challenge DSC is likely limited by poor alignment of small anatomical structures.

This motivated testing a foreground + per-label Dice formulation.

---

# Foreground + Per-Label Dice

## Loss

```python
loss_ct_dice =
    0.3 * foreground_dice
    + 0.7 * per_label_dice
```

where:

```python
per_label_dice =
    mean(Dice(label_i))
```

over all labels present in both fixed and moving images.

---

## Experiment A

```text
lambda_ct_dice = 0.5
```

Results (epoch 51):

| Metric        |  Value |
| ------------- | -----: |
| DSC           | 0.2619 |
| HD95          |  28.14 |
| MTV Error (%) |  13.98 |
| TLG Error (%) |  15.76 |
| NDV (%)       |   1.57 |

### Observation

Performance decreased compared to baseline.

---

## Experiment B

```text
lambda_ct_dice = 1.0
```

Results:

| Metric        |  Value |
| ------------- | -----: |
| DSC           | 0.2721 |
| HD95          |  26.14 |
| MTV Error (%) |  11.95 |
| TLG Error (%) |  11.45 |
| NDV (%)       |   0.80 |

### Observation

Better than the λ=0.5 version but still below the baseline DSC.

### Conclusion

The current foreground + per-label implementation does not improve the challenge score.

A future weighted per-label strategy may still be worth exploring.

---

# PET Soft Lesion Loss

## Motivation

Encourage better alignment of PET lesions using soft lesion masks.

## Observation

The resulting registrations degraded:

* DSC
* MTV
* TLG

No benefit was observed.

## Conclusion

Not retained.

---

# PET Dice Loss

## Motivation

Explicitly supervise lesion alignment.

## Observation

Performance degraded significantly.

## Conclusion

Not retained.

---

# Jacobian Loss

## Motivation

Encourage anatomically plausible deformations and reduce folding.

Implementation:

```python
loss += lambda_jacobian * jacobian_loss
```

Parameters:

```text
lambda_jacobian = 0.001
jacobian_threshold = 0.1
```

## Results

| Metric        |  Value |
| ------------- | -----: |
| DSC           | 0.2459 |
| HD95          |  29.10 |
| MTV Error (%) |  11.36 |
| TLG Error (%) |  10.90 |
| NDV (%)       |   2.22 |

## Visual Inspection

Several training cases showed:

* duplicated vertebral structures
* organ deformation
* anatomically implausible CT warps
* PET distortions

Example observations:

* double vertebral columns
* blurred or stretched anatomical structures
* degraded CT alignment

## Conclusion

The current Jacobian implementation does not improve registration quality and should be abandoned for now.

---

# Increased Smoothness

## Motivation

Reduce excessive local deformations.

Parameters:

```text
lambda_smooth = 0.02
```

instead of:

```text
lambda_smooth = 0.01
```

## Results

| Metric        |  Value |
| ------------- | -----: |
| DSC           | 0.2769 |
| HD95          |  26.86 |
| MTV Error (%) |   5.91 |
| TLG Error (%) |   7.84 |
| NDV (%)       |   0.94 |

## Conclusion

Very similar to baseline.

No significant gain observed.

---

# Reduced PET Weight

## Motivation

Reduce the influence of potentially non-registrable lesion changes.

Parameters:

```text
lambda_pet = 0.1
```

instead of:

```text
lambda_pet = 0.2
```

## Results

| Metric        |  Value |
| ------------- | -----: |
| DSC           | 0.1605 |
| HD95          |  33.12 |
| MTV Error (%) |  15.83 |
| TLG Error (%) |  18.90 |
| NDV (%)       |   0.17 |

## Conclusion

Catastrophic degradation.

The PET channel clearly provides important information for registration.

Future experiments should keep:

```text
lambda_pet = 0.2
```

---

# Current Conclusions

The best-performing model remains:

```text
UNet3D
+ LNCC CT
+ PET MSE
+ Smoothness
+ CT Foreground Dice
```

with:

```text
lambda_pet      = 0.2
lambda_smooth   = 0.01
lambda_ct_dice  = 0.5
```

Current evidence suggests that further improvements are unlikely to come from additional loss terms.

The bottleneck appears to be anatomical representation rather than loss design.

---

# Next Direction: CT Segmentation Pretraining

## Motivation

A large amount of anatomical supervision is currently unused.

Available data:

```text
≈ 560 labeled CT volumes
118 anatomical labels
```

This is substantially more anatomical information than available through registration pairs alone.

## Planned Procedure

### Step 1

Train a CT segmentation network:

```text
CT → 118 labels
```

using all labeled CT volumes, including subjects without follow-up scans.

### Step 2

Transfer encoder weights:

```python
registration_model.encoder.load_state_dict(
    segmentation_model.encoder.state_dict(),
    strict=False,
)
```

### Step 3

Fine-tune the registration model using the standard registration losses.

## Expected Benefit

The encoder should learn stronger anatomical representations before registration training.

This is currently considered the most promising direction for improving DSC.
