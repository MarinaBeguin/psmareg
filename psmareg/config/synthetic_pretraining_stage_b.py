"""Stage B synthetic pretraining recipe, continuing the Stage A checkpoint."""

# A stage recipe inherits every constant of the previous stage.
# pylint: disable-next=wildcard-import,unused-wildcard-import
from psmareg.config.synthetic_pretraining_stage_a import *  # noqa: F403

# Continue Stage A weights with a new loss definition.
START_FROM = "results/pretraining/stage_a/best_val.pt"
LOAD_OPTIMIZER_STATE = False
INHERIT_BEST_VAL_LOSS = False
OUTPUT_DIR = "results/pretraining/stage_b"

# Optimization and architecture
EPOCHS = 120
BATCH_SIZE = 1
LR = 3e-5
MAX_VELOCITY = 12.0
MAX_SCALING = 0.05

# Semantic and regularization losses
LAMBDA_FIELD = 0.15
LAMBDA_STAGE1_FIELD = 0.03
LAMBDA_RESIDUAL_FIELD = 0.05
LAMBDA_EPE = 0.05
LAMBDA_MULTISCALE_FIELD = 0.03
LAMBDA_FIELD_GRADIENT = 0.01
LAMBDA_JACOBIAN = 0.10
LAMBDA_RESIDUAL_BENDING = 0.02
LAMBDA_CT_LABEL_DICE = 0.05
LAMBDA_PET_LABEL_DICE = 0.10
PET_FOREGROUND_WEIGHT = 5.0
LABEL_LOSS_DOWNSAMPLE_FACTOR = 2
MAX_CT_LABEL_CLASSES_PER_BATCH = 4

# PET augmentation
PET_INDEPENDENT_INTENSITY_PROBABILITY = 0.35
PET_INDEPENDENT_SCALE_LOG_STD = 0.15
PET_INDEPENDENT_GAMMA_RANGE = 0.10
PET_INDEPENDENT_BIAS_RANGE = 0.0

# Difficulty schedule
CURRICULUM_TOTAL_EPOCHS = 100
CURRICULUM_EXTRA_HARD_PROBABILITY = 0.15
CURRICULUM_EXTRA_HARD_MAX_SCALE = 1.10
