# Lesion-Preserving Diffeomorphic Registration for Longitudinal PSMA PET/CT

Training, inference and evaluation code for the method described in
[*Lesion-Preserving Diffeomorphic Registration for Longitudinal PSMA PET/CT*](https://www.researchgate.net/publication/415475292_Lesion-Preserving_Diffeomorphic_Registration_for_Longitudinal_PSMA_PETCT),
our contribution to the PSMAReg task of the
[Learn2Reg 2026 challenge](https://www.codabench.org/competitions/15724/#/pages-tab)
(MICCAI 2026).

The model registers a follow-up whole-body PSMA PET/CT scan (moving) to the
baseline scan of the same patient (fixed). It predicts one diffeomorphic
displacement field from both modalities, and is trained so that the PET
lesions keep their volume and uptake when they are warped.

## Contents

1. [Installation](#installation)
2. [Data](#data)
3. [Preparing the manifests](#preparing-the-manifests)
4. [Training](#training)
5. [Inference and evaluation](#inference-and-evaluation)
6. [Configuration](#configuration)
7. [Repository layout](#repository-layout)
8. [Development](#development)
9. [Citation](#citation)

## Installation

The code needs Python 3.10 to 3.12 and a CUDA GPU for training. The published
runs used PyTorch 2.8.0 built for CUDA 12.6.

```bash
git clone https://github.com/MarinaBeguin/psmareg.git
cd psmareg

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# PyTorch for CUDA 12.6. Change "cu126" to match your driver, or skip this
# line to let pip choose the default build.
python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Check the installation:

```bash
python3 -c "import torch, psmareg; print(torch.__version__, torch.cuda.is_available())"
```

All commands below are run from the repository root with the virtual
environment active.

## Data

### PSMAReg challenge data

The method was developed on the training and validation data of the PSMAReg
task. Register on the
[challenge page](https://www.codabench.org/competitions/15724/#/pages-tab) to
download the dataset; the same page describes the data, the evaluation metrics
and the submission format.

In short, the dataset is derived from the public PSMA-PET-CT-Lesions cohort
(autoPET, LMU University Hospital Munich): 597 PSMA PET/CT studies from 378
patients, of whom 164 have at least one follow-up scan. Every study is
resampled to a common grid of 192 × 192 × 288 voxels at
2.7344 × 2.7344 × 3.27 mm. The hidden test set is not distributed.

The download contains:

| Item | Content |
| --- | --- |
| `imagesTr/` | CT and PET volumes, named `<case>_<modality>_<timepoint>.nii.gz` with modality `0000` = CT, `0001` = PSMA PET, and timepoint `00` = baseline, `01`, `02`, ... = follow-ups |
| `labelsTr/` | One label map per image, with the same filename as the image it annotates |
| `PSMAReg_dataset.json` | The training and validation registration pairs |

### Using your own data

The code reads four volumes per registration pair, plus labels for training.
Your data must meet these requirements:

| Volume | Format | Content |
| --- | --- | --- |
| CT | NIfTI (`.nii.gz`), 3D | Hounsfield units. The loader removes the scanner table and windows the values to [-300, 300] HU |
| PET | NIfTI, 3D | PSMA uptake in any positive unit (SUV or Bq/mL). Each scan is scaled by its own 99th percentile |
| CT label | NIfTI, 3D, integer | [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) `total` classes, 0 = background, 1 to 117 = structures |
| PET label | NIfTI, 3D, integer | Lesion mask, 0 = background, any positive value = lesion |

- The CT, the PET and both label maps of a scan share one voxel grid, and the
  baseline and follow-up scans of a pair have the same array shape. The code
  works in voxel coordinates and does not resample.
- The models were trained and tested only on the challenge grid
  (192 × 192 × 288 voxels). Resample your data to this grid to reuse the
  recipes unchanged.
- Labels are needed for the scans used in training and validation. Inference
  needs only the four images.
- Keep the patients of the validation set out of both the fine-tuning and the
  pretraining training sets.

## Preparing the manifests

Training reads CSV manifests that list absolute file paths. Regenerate them if
the dataset moves.

### Paired manifests (fine-tuning, validation, inference)

```bash
python3 scripts/build_paired_data.py \
  --json_path /path/to/dataset/PSMAReg_dataset.json \
  --dataset_root /path/to/dataset \
  --output_dir manifests/paired
```

This writes `manifests/paired/train_pairs.csv` and
`manifests/paired/val_pairs.csv`, one row per baseline/follow-up pair.

With your own data, write the two CSV files yourself with these columns:

| Column | Required | Content |
| --- | --- | --- |
| `subject` | yes | Patient identifier |
| `followup_index` | yes | Follow-up number of the moving scan (`01`, `02`, ...) |
| `fixed_ct`, `fixed_pet` | yes | Baseline images |
| `moving_ct`, `moving_pet` | yes | Follow-up images |
| `fixed_ct_label`, `moving_ct_label` | for training and validation | CT label maps |
| `fixed_pet_label`, `moving_pet_label` | for training and validation | PET lesion masks |

### Single-scan manifests (synthetic pretraining)

Pretraining does not use pairs: it deforms single scans synthetically. Build
the list of scans, then split it by patient:

```bash
python3 scripts/create_pretraining_manifest.py \
  --images_dir /path/to/dataset/imagesTr \
  --labels_dir /path/to/dataset/labelsTr \
  --output_csv manifests_pretraining/all_scans.csv

python3 scripts/split_pretraining_manifest.py \
  --input_manifest manifests_pretraining/all_scans.csv \
  --output_dir manifests_pretraining
```

This writes `manifests_pretraining/training_data.csv` and
`manifests_pretraining/validation_data.csv` (80 % / 20 % of the patients, seed
42). The pretraining recipes also drop every patient listed in
`manifests/paired/val_pairs.csv`, so build the paired manifests first.

With your own data, the columns are `subject`, `ct`, `pet`, `ct_label` and
`pet_label`.

### Preprocessing cache (optional)

Preprocessed volumes are cached on disk the first time they are read, so no
step is required here. To fill the paired cache before training instead of
during the first epoch:

```bash
python3 scripts/warm_paired_cache.py \
  --manifest manifests/paired/train_pairs.csv \
  --manifest manifests/paired/val_pairs.csv
```

The caches are written to `./psmareg_paired_cache` and
`./psmareg_synthetic_cache`. To put them on a larger disk, set these variables
before running any command:

```bash
export PSMAREG_PAIRED_CACHE_DIR=/scratch/psmareg_paired_cache
export PSMAREG_SYNTHETIC_CACHE_DIR=/scratch/psmareg_synthetic_cache
```

## Training

Training has four runs. Stages A, B and C learn from synthetic deformations
of single scans; the final run fine-tunes on real baseline/follow-up pairs.

```text
Stage A ──► Stage B ──────────┐
                              ├──► Fine-tuning
Stage C ──────────────────────┘
```

| Run | Model | Starts from | Epochs | Output |
| --- | --- | --- | ---: | --- |
| Stage A | Deformable cascade | Random weights | 80 | `results/pretraining/stage_a` |
| Stage B | Deformable cascade | Stage A `best_val.pt` | 120 | `results/pretraining/stage_b` |
| Stage C | Affine + deformable cascade | Random weights | 160 | `results/pretraining/stage_c` |
| Fine-tuning | Affine + deformable cascade | Stage C, with the refiner weights of Stage B | 250 | `results/fine_tuning` |

Stage C does not depend on Stages A and B, so it can run at the same time on
a second GPU. Fine-tuning needs both Stage B and Stage C to be finished.

### Commands

```bash
# 1. Stage A
python3 train_synthetic_registration_pretraining.py \
  --config psmareg/config/synthetic_pretraining_stage_a.py

# 2. Stage B (after Stage A)
python3 train_synthetic_registration_pretraining.py \
  --config psmareg/config/synthetic_pretraining_stage_b.py

# 3. Stage C (any time; independent of Stages A and B)
python3 train_synthetic_registration_pretraining.py \
  --config psmareg/config/synthetic_pretraining_stage_c.py

# 4. Fine-tuning (after Stages B and C)
python3 train_fine_tuning.py --config psmareg/config/fine_tuning_final.py
```

Each recipe already points to the manifests and checkpoints produced by the
previous steps, so no path needs to be edited when the default locations are
used.

### Launchers

`scripts/local/` wraps the same four commands. A launcher selects the GPU,
runs from the repository root and copies the output to
`logs/<run>_<timestamp>.log`:

```bash
GPU=0 scripts/local/run_pretraining_stage_a.sh
GPU=0 scripts/local/run_pretraining_stage_b.sh
GPU=1 scripts/local/run_pretraining_stage_c.sh
GPU=0 scripts/local/run_fine_tuning_final.sh
```

Set `DRY_RUN=1` to print the command without running it. The header of each
launcher lists its other variables (`OUTPUT_DIR`, `EPOCHS`, `NUM_WORKERS`,
...).

### Outputs and resuming

Every run directory contains `metrics.csv` (one row per epoch), `latest.pt`
and `best_val.pt`. Fine-tuning also keeps the best checkpoint for three
challenge-like validation metrics:

| Checkpoint | Selected on |
| --- | --- |
| `best_val.pt` | Validation selection score |
| `best_challenge_ct_dsc.pt` | CT organ Dice |
| `best_challenge_pet_dsc.pt` | PET lesion Dice |
| `best_challenge_pet_biomarker.pt` | PET biomarker error (MTV and TLG) |

The results of the paper and our challenge submission use
`best_challenge_pet_biomarker.pt`.

To continue an interrupted fine-tuning run:

```bash
python3 train_fine_tuning.py \
  --config psmareg/config/fine_tuning_final.py \
  --resume_from results/fine_tuning/latest.pt
```

To continue an interrupted pretraining stage, with Stage B as the example:

```bash
START_FROM=results/pretraining/stage_b/latest.pt \
  scripts/local/run_pretraining_stage_b.sh
```

Use a new `--output_dir` for a new experiment so that a previous run is not
overwritten.

## Inference and evaluation

### Displacement fields

```bash
python3 scripts/export_displacements.py \
  --checkpoint results/fine_tuning/best_challenge_pet_biomarker.pt \
  --manifest manifests/paired/val_pairs.csv \
  --output_dir results/predictions
```

This writes one `disp_<subject>_00_<subject>_<followup>.nii.gz` per pair: a
full-resolution field of shape `[X, Y, Z, 3]`, in voxels, defined on the fixed
grid and pulling the moving scan onto it. The manifest needs only the image
columns. Add `--lowres` to write the half-resolution fields of the validation
submission format.

### Challenge metrics

Local scoring uses the official scoring program, which is not included here.
Clone it into the repository root, then score half-resolution fields:

```bash
git clone https://github.com/JHU-MedImage-Reg/Learn2Reg26_PSMA.git

python3 scripts/export_displacements.py \
  --checkpoint results/fine_tuning/best_challenge_pet_biomarker.pt \
  --manifest manifests/paired/val_pairs.csv \
  --output_dir results/predictions_lowres \
  --lowres

python3 scripts/score_displacements.py \
  --reference_dir /path/to/dataset \
  --prediction_dir results/predictions_lowres \
  --output_dir results/scores
```

`--reference_dir` is the dataset root that contains `PSMAReg_dataset.json`.

### Figures

```bash
# Registration panels for the validation subjects
CHECKPOINT=results/fine_tuning/best_challenge_pet_biomarker.pt \
  scripts/local/run_visualize_val_registration_grid.sh

# Qualitative figure of the paper
python3 scripts/make_results_qualitative_figure.py \
  --checkpoint results/fine_tuning/best_challenge_pet_biomarker.pt \
  --manifest manifests/paired/val_pairs.csv \
  --output_prefix results/figures/results_qualitative
```

Both need the labeled validation manifest.

## Configuration

Each run is defined by a Python recipe made of uppercase constants:

| Recipe | Run |
| --- | --- |
| [synthetic_pretraining_stage_a.py](psmareg/config/synthetic_pretraining_stage_a.py) | Stage A |
| [synthetic_pretraining_stage_b.py](psmareg/config/synthetic_pretraining_stage_b.py) | Stage B |
| [synthetic_pretraining_stage_c.py](psmareg/config/synthetic_pretraining_stage_c.py) | Stage C |
| [fine_tuning_final.py](psmareg/config/fine_tuning_final.py) | Fine-tuning |

To define a new experiment, copy a recipe and pass the copy to `--config`.
[PARAMETERS.md](psmareg/config/PARAMETERS.md) documents every parameter.

The command line overrides the paths and the number of epochs directly, and
any other parameter with `--set NAME=VALUE`:

```bash
python3 train_fine_tuning.py \
  --config psmareg/config/fine_tuning_final.py \
  --output_dir results/fine_tuning_test \
  --epochs 2 \
  --set num_workers=4
```

Booleans are written `true` or `false`, tuples as JSON lists
(`--set multiscale_corr_scales=[2,4,8]`), and `none` unsets an optional path
(`--set pretrained_checkpoint=none`). Unknown names and out-of-range values
are rejected before any data is loaded.

## Repository layout

```text
train_synthetic_registration_pretraining.py   Stages A, B and C
train_fine_tuning.py                          Fine-tuning on real pairs
psmareg/
  config/       Recipes, typed configuration and parameter reference
  data/         Manifests, NIfTI reading, preprocessing, caches, augmentation,
                synthetic deformations
  models/       Registration networks, pre-alignment and losses
  training/     Epoch loops, checkpoints, model selection, metric logs
  evaluation/   Challenge metrics, displacement export, figures
scripts/        Manifest building, cache, inference, scoring, figures
scripts/local/  Launchers for a local GPU
tests/          Unit tests
```

## Development

```bash
python3 -m pip install -r requirements-dev.txt

python3 -m pytest tests
yapf -i -r psmareg scripts tests train_*.py
pylint psmareg scripts tests train_*.py
mypy psmareg scripts train_*.py
```

The code follows the Google Python style guide (yapf `google` preset, 80
columns, Google docstrings). The tool settings are in `pyproject.toml`.

## Citation

If you use this code, please cite:

```bibtex
@misc{beguin2026lesionpreserving,
  title  = {Lesion-Preserving Diffeomorphic Registration for Longitudinal
            {PSMA} {PET/CT}},
  author = {Beguin, Marina and others},
  year   = {2026},
  note   = {Learn2Reg 2026 (MICCAI) method description, peer-reviewed on
            OpenReview},
  url    = {https://www.researchgate.net/publication/415475292_Lesion-Preserving_Diffeomorphic_Registration_for_Longitudinal_PSMA_PETCT}
}
```

The journal reference will replace this entry once the article is published.

If you use the PSMAReg data, please also cite the dataset and the challenge,
as required by the
[challenge terms](https://www.codabench.org/competitions/15724/#/pages-tab):

- Jeblick, K., et al. *A whole-body PSMA-PET/CT dataset with manually
  annotated tumor lesions (PSMA-PET-CT-Lesions)*, version 1. The Cancer
  Imaging Archive, 2024. https://doi.org/10.7937/r7ep-3x37
- Hering, A., et al. *Learn2Reg: comprehensive multi-task medical image
  registration challenge, dataset and evaluation in the era of deep
  learning*. IEEE Transactions on Medical Imaging, 42(3), 2023.
