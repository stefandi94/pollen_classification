import os
import os.path as osp

RANDOM_STATE = 42
NUM_OF_CLASSES = 50
# DIRECTORIES
# =======================================================================================================================

# BASE DIRECTORY
BASE_DIR = os.path.dirname(osp.realpath(__file__))

# DATA DIRECTORY
DATA_DIR = osp.join(BASE_DIR, 'data')

TRAINING_DIR = osp.join(DATA_DIR, 'training')
VALID_DIR = osp.join(DATA_DIR, 'valid')
TEST_DIR = osp.join(DATA_DIR, 'test')

# MODEL_DIR
MODEL_DIR = osp.join(BASE_DIR, 'model_weights')

########################################################################################################################

# MODEL FOLDERS

# RESIDUAL_DIR
RESIDUAL_DIR = osp.join(MODEL_DIR, 'residual_net')
RESIDUAL_18 = osp.join(RESIDUAL_DIR, 'resnet_18')

# WIDE RESIDUALS
WIDE_RESIDUAL_DIR = osp.join(MODEL_DIR, 'wide_residuals')

# VGG DIR
VGG_DIR = osp.join(MODEL_DIR, 'vgg_net')

# INCEPTION DIR
INCEPTION_DIR = osp.join(MODEL_DIR, 'inception_net')

KERNEL_REGULARIZER = 0.0000
ACTIVITY_REGULARIZER = 0.0000
BIAS_REGULARIZER = 0.0000
