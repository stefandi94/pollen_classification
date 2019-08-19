import os.path as osp

RANDOM_STATE = 42
NUM_OF_CLASSES = 50
# DIRECTORIES
# =======================================================================================================================

# BASE DIRECTORY
BASE_DIR = osp.dirname(osp.realpath(__file__))

RAW_DATA_DIR = osp.join(BASE_DIR, 'raw_data')
# DATA DIRECTORY
DATA_DIR = osp.join(BASE_DIR, 'data')

TRAIN_DIR = osp.join(DATA_DIR, 'training')
VALID_DIR = osp.join(DATA_DIR, 'valid')
TEST_DIR = osp.join(DATA_DIR, 'test')

NORMALIZED_TRAIN_DIR = osp.join(TRAIN_DIR, 'normalized_data')
STANDARDIZED_TRAIN_DIR = osp.join(TRAIN_DIR, 'standardized_data')

NORMALIZED_VALID_DIR = osp.join(VALID_DIR, 'normalized_data')
STANDARDIZED_VALID_DIR = osp.join(VALID_DIR, 'standardized_data')

NORMALIZED_TEST_DIR = osp.join(TEST_DIR, 'normalized_data')
STANDARDIZED_TEST_DIR = osp.join(TEST_DIR, 'standardized_data')

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
