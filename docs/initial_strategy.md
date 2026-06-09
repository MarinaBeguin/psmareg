# Learn2Reg 2026 – PSMAReg

## Dataset analysis and initial registration strategy

### Challenge objective

The PSMAReg task focuses on longitudinal registration of whole-body PSMA PET/CT scans acquired before and after therapy.

The objective is to estimate a deformable displacement field that aligns each follow-up examination (moving image) to the corresponding baseline examination (fixed image).

For each patient:

* Baseline scan = fixed image
* Follow-up scan = moving image
* Output = dense deformation field mapping follow-up to baseline

The challenge evaluates registration quality through propagated labels and image-based metrics.

---

## Dataset structure

The dataset is organized into three folders:

```text
imagesTr/
labelsTr/
imagesVal/
```

The image filenames follow the convention:

```text
PSMARegPSMA_<subject>_<modality>_<timepoint>.nii.gz
```

where:

* `<subject>` identifies the patient
* `<modality>`:

  * `0000` = CT
  * `0001` = PSMA PET
* `<timepoint>`:

  * `00` = baseline
  * `01`, `02`, ... = follow-up examinations

Examples:

```text
PSMARegPSMA_0006_0000_00
```

Baseline CT.

```text
PSMARegPSMA_0006_0001_00
```

Baseline PET.

```text
PSMARegPSMA_0006_0000_01
```

Follow-up CT.

```text
PSMARegPSMA_0006_0001_01
```

Follow-up PET.

---

## Paired and unpaired subjects

The challenge contains two types of training cases.

### Paired subjects

These subjects contain:

```text
baseline CT
baseline PET
follow-up CT
follow-up PET
```

or several follow-up examinations.

Example:

```text
0000 -> baseline
0001 -> follow-up 1
0002 -> follow-up 2
```

For these subjects, one registration pair is created for each follow-up:

```text
baseline <- follow-up 1
baseline <- follow-up 2
```

These cases constitute the main supervision signal for registration training.

### Unpaired subjects

Some subjects contain only:

```text
baseline CT
baseline PET
```

without any follow-up examination.

Example:

```text
PSMARegPSMA_0123_0000_00
PSMARegPSMA_0123_0001_00
```

These subjects cannot be used directly for supervised registration because no moving image exists.

However, they may be useful for:

* self-supervised pretraining
* representation learning
* modality-specific feature learning

---

## Labels

The training set also provides labels:

### CT labels

```text
0000 = organ segmentation
```

Generated automatically using TotalSegmentator.

### PET labels

```text
0001 = lesion segmentation
```

Manually annotated by radiologists.

These labels can potentially be used as additional supervision during training.

---

## Interpretation of PET/CT scans

A key question was whether CT and PET should be treated as independent registration targets.

After reviewing the dataset description and acquisition protocol, the most plausible interpretation is that each PET/CT examination represents a single multimodal acquisition.

Therefore:

```text
CT_baseline
PET_baseline
```

describe the same patient at the same acquisition time.

Similarly:

```text
CT_followup
PET_followup
```

describe the same patient at the follow-up visit.

Consequently, registration should estimate a single anatomical deformation field:

```text
follow-up -> baseline
```

which is shared across both modalities.

The goal is not to register CT to PET.

Instead, the goal is to register:

```text
CT_followup  -> CT_baseline
PET_followup -> PET_baseline
```

using the same displacement field.

---

## Initial modelling strategy

### Phase 1: Multimodal PET/CT registration

Once the CT-only baseline is operational, PET information will be incorporated.

Inputs:

```text
CT_baseline
PET_baseline
CT_followup
PET_followup
```

Output:

```text
dense displacement field
```

Training loss:

```text
L = L_CT + λ_pet L_PET + λ_reg L_smooth
```

where:

* L_CT measures CT alignment
* L_PET measures PET alignment
* L_smooth regularizes the deformation

The CT term is expected to dominate because:

* CT contains stable anatomical information
* PET uptake can change substantially after therapy
* lesions may appear, disappear or shrink

PET should therefore provide complementary information rather than drive the registration.


## Future possible extensions

Potential future developments include:

1. Incorporating organ labels into the loss function.
2. Incorporating lesion labels into the loss function.
3. Learning multimodal latent representations using Point Transformers.
4. Exploring Gaussian-based anatomical representations for registration.
5. Using unpaired subjects for self-supervised pretraining.

