"""Stage A synthetic pretraining recipe.

Edit a copy for new experiments. Defaults for omitted parameters are declared
and validated in :mod:`psmareg.config.synthetic_pretraining_config`.
"""

import os

# Data and initialization
TRAIN_MANIFEST = "manifests_pretraining/training_data.csv"
VAL_MANIFEST = "manifests_pretraining/validation_data.csv"
EXCLUDE_SUBJECTS_MANIFEST = "manifests/paired/val_pairs.csv"
CACHE_DIR = os.environ.get(
    "PSMAREG_SYNTHETIC_CACHE_DIR",
    "./psmareg_synthetic_cache",
)
OUTPUT_DIR = "results/pretraining/stage_a"
START_FROM: str | None = None

# Architecture and optimization
MODEL = "transmorph_v2_cascade"
EPOCHS = 80
BATCH_SIZE = 2
LR = 5e-5
NUM_WORKERS = 4
PATIENCE = 20
AMP = True
VALIDATION_INTERVAL = 5
MAX_VELOCITY = 10.0
MAX_VELOCITY_START: float | None = 8.0
MAX_VELOCITY_RAMP_EPOCHS = 10
MAX_RESIDUAL_DISP = 4.0

# Image and field losses
SIMILARITY_DOWNSAMPLE_FACTOR = 2
VALIDATION_SIMILARITY_DOWNSAMPLE_FACTOR = 1
LAMBDA_FIELD = 0.08
LAMBDA_STAGE1_FIELD = 0.015
LAMBDA_RESIDUAL_FIELD = 0.03
LAMBDA_SMOOTH = 0.01
LAMBDA_EPE = 0.02
LAMBDA_MULTISCALE_FIELD = 0.01
LAMBDA_FIELD_GRADIENT = 0.005
LAMBDA_JACOBIAN = 0.03
JACOBIAN_MARGIN = 0.05
JACOBIAN_LOSS_STRIDE = 4

# Synthetic target generation
TARGET_MIN_JACOBIAN = 0.01
TARGET_GENERATION_ATTEMPTS = 8
COARSE_TARGET_DOWNSAMPLE_FACTOR = 4
MAX_ELASTIC_DISPLACEMENT = 6.0
MAX_TRANSLATION = 6.0
MAX_ROTATION_DEGREES = 4.0
MAX_SCALING = 0.04

# Difficulty schedule
CURRICULUM_TOTAL_EPOCHS = 80
CURRICULUM_MIN_SCALE = 0.25
CURRICULUM_WARMUP_FRACTION = 0.20
CURRICULUM_RAMP_FRACTION = 0.50
CURRICULUM_EASY_PROBABILITY = 0.25
CURRICULUM_EXTRA_HARD_PROBABILITY = 0.05
CURRICULUM_EXTRA_HARD_MAX_SCALE = 1.0
