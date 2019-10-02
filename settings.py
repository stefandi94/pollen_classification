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

WEIGHTS_DIR = osp.join(BASE_DIR, 'new_weights')
NS_WEIGHTS_DIR = osp.join(WEIGHTS_DIR, 'ns')
OS_WEIGHTS_DIR = osp.join(WEIGHTS_DIR, 'os')

NS_NORM_WEIGHTS_DIR = osp.join(NS_WEIGHTS_DIR, 'normalized')
NS_STAND_WEIGHTS_DIR = osp.join(NS_WEIGHTS_DIR, 'standard_normal')

OS_NORM_WEIGHTS_DIR = osp.join(OS_WEIGHTS_DIR, 'normalized')
OS_STAND_WEIGHTS_DIR = osp.join(OS_WEIGHTS_DIR, 'standard_normal')

MODEL_DIR = osp.join(BASE_DIR, 'model_weights')

########################################################################################################################
LIFE_STAT_COMP = osp.join(NS_TRAIN_DIR, '/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/NS/train/')
SCATTER_STAT_COMP = osp.join('/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/NS/train/')
SIZE_STAT_COMP = osp.join('/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/NS/train/')
# MODEL FOLDERS

KERNEL_REGULARIZER = 0.0000
ACTIVITY_REGULARIZER = 0.0000
BIAS_REGULARIZER = 0.0000
