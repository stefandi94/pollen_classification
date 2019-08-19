import os
import os.path as osp
import pickle

import numpy as np

from settings import DATA_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, RANDOM_STATE

np.random.seed(RANDOM_STATE)


def load_data(data_path, filename):
    # file = open(data_path, 'rb')
    # data = pickle.load(file)
    # file.close()
    with open(osp.join(data_path, f'{filename}.pckl'), 'rb') as handle:
        data = pickle.load(handle)
    return data


def save_data(file, data_path, filename):
    # f = open(osp.join(data_path, f'{filename}.pckl'), 'wb')
    # pickle.dump(file, f)
    # f.close()
    with open(osp.join(data_path, f'{filename}.pckl'), 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_train_valid_test_data(data: list,
                                 labels: list,
                                 train_size: float = 0.75,
                                 valid_size: float = 0.1,
                                 test_size: float = 0.15):
    """
        Given data and labels, split it into train,
        valid and test data and labels and then save it.
    """

    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)

    # shuffle data and labels
    for index in range(len(data)):
        data[index] = np.array(data[index])[indices]
    labels = np.array(labels)[indices]

    train_indices = (0, int(train_size * len(data[0])))
    valid_indices = (train_indices[-1], train_indices[-1] + int(valid_size * len(data[0])))
    test_indices = (valid_indices[-1], valid_indices[-1] + int(test_size * len(data[0])))

    train_data = []
    valid_data = []
    test_data = []

    for feature in data:
        train_data.append([np.array(feature)[train_indices[0]: train_indices[1]]])
        valid_data.append([np.array(feature)[valid_indices[0]: valid_indices[1]]])
        test_data.append([np.array(feature)[test_indices[0]: test_indices[1]]])

    train_labels = [np.array(labels)[train_indices[0]: train_indices[1]]]
    valid_labels = [np.array(labels)[valid_indices[0]: valid_indices[1]]]
    test_labels = [np.array(labels)[test_indices[0]: test_indices[1]]]

    feature_names = ['scatter', 'size', 'life_1', 'spectrum', 'life_2']
    data_to_save = [train_data, valid_data, test_data]
    labels_to_save = [train_labels, valid_labels, test_labels]

    dirs_to_save = [TRAIN_DIR, VALID_DIR, TEST_DIR]
    for dir_index, dir in enumerate(dirs_to_save):
        if not osp.exists(dir):
            os.makedirs(dir)

        for feature_index, feature in enumerate(feature_names):
            save_data(file=data_to_save[dir_index][feature_index][0],
                      data_path=dir,
                      filename=feature)

        save_data(file=labels_to_save[dir_index][0],
                  data_path=dir,
                  filename="target")


def flatten_data(data):
    new_train_data = np.zeros((len(data[0]), 2629))
    for vector_index in range(len(data[0])):
        new_vector = []
        for feature_index, vector in enumerate(data):
            if feature_index == 1:
                new_vector.extend([vector[vector_index]])
            else:
                new_vector.extend(np.array(vector[vector_index]).flatten())
        new_train_data[vector_index] = new_vector
    return new_train_data


def find_statistical_components(data):
    min_value = min(np.array(data).flatten())
    max_value = max(np.array(data).flatten())
    mean_value = np.mean(np.array(data).flatten())
    std_value = np.std(np.array(data).flatten())

    return {'min': min_value, 'max': max_value, 'mean': mean_value, 'std': std_value}


def normalize_data(data_to_normalize, mean_value, std_value):
    data_to_normalize -= mean_value
    data_to_normalize /= std_value
    return data_to_normalize


def standardize_data(data_to_standardize, max_value, min_value):
    data_to_standardize -= min_value
    data_to_standardize /= max_value - min_value
    return data_to_standardize


if __name__ == '__main__':
    # path = '/mnt/hdd/data/'

    feature_to_load = ['scatter', 'size', 'life_1', 'spectrum', 'life_2']
    data_to_load = [TRAIN_DIR, VALID_DIR, TEST_DIR]

    for feature in feature_to_load:
        with open(osp.join(DATA_DIR, f'{feature}_statistical_components.pckl'), 'rb') as handle:
            stat_components = pickle.load(handle)

        for dir_type in data_to_load:
            data = load_data(dir_type, feature)

            normalized_data = normalize_data(data, stat_components['mean'], stat_components['std'])
            standardized_data = standardize_data(data, stat_components['max'], stat_components['min'])

            normalized_path = osp.join(dir_type, 'normalized_data')
            standardized_path = osp.join(dir_type, 'standardized_data')

            if not osp.exists(normalized_path):
                os.makedirs(normalized_path)
            if not osp.exists(standardized_path):
                os.makedirs(standardized_path)

            with open(osp.join(normalized_path, f'{feature}.pckl'), 'wb') as handle:
                pickle.dump(normalized_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(osp.join(standardized_path, f'{feature}.pckl'), 'wb') as handle:
                pickle.dump(standardized_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print()

        # data = load_data(osp.join(TRAIN_DIR, f'{feature}.pckl'))
        # dict_with_stat_comp = find_statistical_components(data)
        #
        # with open(osp.join(DATA_DIR, f'{feature}_statistical_components.pckl'), 'wb') as handle:
        #     pickle.dump(dict_with_stat_comp, handle, protocol=pickle.HIGHEST_PROTOCOL)
