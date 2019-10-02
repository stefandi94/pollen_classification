import os
import pickle

from utils.utilites import count_values

import numpy as np
from keras.utils import to_categorical

from settings import OS_TRAIN_DIR, NS_TRAIN_DIR, NS_DATA_DIR, OS_DATA_DIR
from source.get_model import get_model
from source.models import ANN
from utils.converting_raw_data import transform_raw_data
from utils.split_data import load_data, convert_data_to_normal_0_1, convert_data_to_standard_normal

data_path = './'
data, labels, class_to_num, feature_names = transform_raw_data('/mnt/hdd/PycharmProjects/pollen_classification/data/raw/OS/')
model_name = 'ANN'
NUM_OF_CLASSES = 8
normalize = True
NS = False

load_dir = '/mnt/hdd/PycharmProjects/pollen_classification/new_weights/os/standard_normal/smooth_factor_0.0/'\
           'optimizer_adam/learning_rate_type_cosine/model_name_ANN/8/25-0.872-0.713-0.949-0.694.hdf5'

# ova ima 10 clase
# load_dir = '/mnt/hdd/PycharmProjects/pollen_classification/new_weights/os/standardized/smooth_factor_0.0/' \
#            'optimizer_rmsprop/learning_rate_type_cyclic/model_name_GRU/5-1.100-0.606-0.808-0.717.hdf5'

parameters = {'epochs': 30,
              'batch_size': 256,
              'optimizer': 'adam',
              'num_classes': NUM_OF_CLASSES,
              'save_dir': f'./save_dir',
              'load_dir': f'{load_dir}'}

if NS:
    # load statistical components
    life_1_stat_comp = load_data(NS_TRAIN_DIR, 'life_1_stat_comp')
    scatter_stat_comp = load_data(NS_TRAIN_DIR, 'scatter_stat_comp')
    spectrum_stat_comp = load_data(NS_TRAIN_DIR, 'size_stat_comp')

    with open(os.path.join(NS_TRAIN_DIR, 'normalized_data', f'dict_mapping_{NUM_OF_CLASSES}.pckl'), 'rb') as handle:
        dict_mapping = pickle.load(handle)

    with open(os.path.join(NS_DATA_DIR, 'label_to_index.pckl'), 'rb') as handle:
        labels_pickle_mapping = pickle.load(handle)

else:
    life_1_stat_comp = load_data(OS_TRAIN_DIR, 'life_1_stat_comp')
    scatter_stat_comp = load_data(OS_TRAIN_DIR, 'scatter_stat_comp')
    spectrum_stat_comp = load_data(OS_TRAIN_DIR, 'size_stat_comp')

    with open(os.path.join(OS_TRAIN_DIR, 'normalized_data', f'dict_mapping_{NUM_OF_CLASSES}.pckl'), 'rb') as handle:
        dict_mapping = pickle.load(handle)

    with open(os.path.join(OS_DATA_DIR, 'label_to_index.pckl'), 'rb') as handle:
        labels_pickle_mapping = pickle.load(handle)

print()

new_data = [[], [], []]
stat_components = [scatter_stat_comp, life_1_stat_comp, spectrum_stat_comp]
for index, feature in enumerate(['scatter', 'life_1', 'spectrum']):
    if normalize:
        new_data[index] = convert_data_to_standard_normal(data[feature],
                                                          stat_components[index]['mean'],
                                                          stat_components[index]['std'])
    else:
        new_data[index] = convert_data_to_normal_0_1(data[feature],
                                                     stat_components[index]['min'],
                                                     stat_components[index]['max'])

# convert to same labels as model was trained
# num_to_class = dict((label, clas) for (clas, label) in labels_pickle_mapping.items())

# label_class = [class_to_num[label] for label in ]
new_labels = [dict_mapping[label] for label in labels]

# new_labels = [labels_pickle_mapping[label] for label in new_labels]
cate_label = to_categorical(new_labels, num_classes=NUM_OF_CLASSES)

dl_model = ANN()
dl_model.load_model(parameters['load_dir'])

# y_pred = dl_model.predict(data)
test_acc = dl_model.model.evaluate(new_data, cate_label, batch_size=256)

print(f'Accuracy is: {test_acc[1]}')
