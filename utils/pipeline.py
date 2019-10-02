import os
import pickle

from utils.converting_raw_data import transform_raw_data
from utils.split_data import load_data, standardize_data, normalize_data

data_path = './'
data, labels, class_to_num, feature_names = transform_raw_data('/mnt/hdd/PycharmProjects/pollen_classification/data/raw/test/')

normalize = True
NS = True

# load statistical components
life_1_stat_comp = load_data('/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/NS/train/', 'life_1_stat_comp')
scatter_stat_comp = load_data('/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/NS/train/', 'scatter_stat_comp')
size_stat_comp = load_data('/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/NS/train/', 'size_stat_comp')

if NS:
    with open(os.path.join('/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/NS/', 'label_to_index.pckl'), 'rb') as handle:
        labels_pickle_mapping = pickle.load(handle)
else:
    with open(os.path.join('/mnt/hdd/PycharmProjects/pollen_classification/data/extracted/OS/', 'label_to_index.pckl'), 'rb') as handle:
        labels_pickle_mapping = pickle.load(handle)

print()

if normalize:
    data[0] = normalize_data(data[0], life_1_stat_comp['mean_value'], life_1_stat_comp['std_value'])
    data[1] = normalize_data(data[1], scatter_stat_comp['mean_value'], scatter_stat_comp['std_value'])
    data[2] = normalize_data(data[2], size_stat_comp['mean_value'], size_stat_comp['std_value'])

else:
    data[0] = standardize_data(data[0], life_1_stat_comp['min_value'], life_1_stat_comp['max_value'])
    data[1] = standardize_data(data[1], scatter_stat_comp['min_value'], scatter_stat_comp['max_value'])
    data[2] = standardize_data(data[2], size_stat_comp['min_value'], size_stat_comp['max_value'])



