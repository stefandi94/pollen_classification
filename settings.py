import os.path as osp

RANDOM_STATE = 42
NUM_OF_CLASSES = 50

# DIRECTORIES
# =======================================================================================================================

# BASE DIRECTORY
BASE_DIR = osp.dirname(osp.realpath(__file__))

DATA_DIR = osp.join(BASE_DIR, 'data')

RAW_DATA_DIR = osp.join(DATA_DIR, 'raw')
OS_RAW_DATA_DIR = osp.join(RAW_DATA_DIR, 'OS')
NS_RAW_DATA_DIR = osp.join(RAW_DATA_DIR, 'NS')

EXTRACTED_DATA_DIR = osp.join(DATA_DIR, 'extracted')
OS_DATA_DIR = osp.join(EXTRACTED_DATA_DIR, 'OS')
NS_DATA_DIR = osp.join(EXTRACTED_DATA_DIR, 'NS')

OS_TRAIN_DIR = osp.join(OS_DATA_DIR, 'train')
OS_VALID_DIR = osp.join(OS_DATA_DIR, 'valid')
OS_TEST_DIR = osp.join(OS_DATA_DIR, 'test')

NS_TRAIN_DIR = osp.join(NS_DATA_DIR, 'train')
NS_VALID_DIR = osp.join(NS_DATA_DIR, 'valid')
NS_TEST_DIR = osp.join(NS_DATA_DIR, 'test')

OS_NORMALIZED_TRAIN_DIR = osp.join(OS_TRAIN_DIR, 'normalized_data')
OS_STANDARDIZED_TRAIN_DIR = osp.join(OS_TRAIN_DIR, 'standardized_data')

OS_NORMALIZED_VALID_DIR = osp.join(OS_VALID_DIR, 'normalized_data')
OS_STANDARDIZED_VALID_DIR = osp.join(OS_VALID_DIR, 'standardized_data')

OS_NORMALIZED_TEST_DIR = osp.join(OS_TEST_DIR, 'normalized_data')
OS_STANDARDIZED_TEST_DIR = osp.join(OS_TEST_DIR, 'standardized_data')

NS_NORMALIZED_TRAIN_DIR = osp.join(NS_TRAIN_DIR, 'normalized_data')
NS_STANDARDIZED_TRAIN_DIR = osp.join(NS_TRAIN_DIR, 'standardized_data')

NS_NORMALIZED_VALID_DIR = osp.join(NS_VALID_DIR, 'normalized_data')
NS_STANDARDIZED_VALID_DIR = osp.join(NS_VALID_DIR, 'standardized_data')

NS_NORMALIZED_TEST_DIR = osp.join(NS_TEST_DIR, 'normalized_data')
NS_STANDARDIZED_TEST_DIR = osp.join(NS_TEST_DIR, 'standardized_data')

WEIGHTS_DIR = osp.join(BASE_DIR, 'model_weights')
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
