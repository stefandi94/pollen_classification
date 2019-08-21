import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np

from settings import RANDOM_STATE, NS_RAW_DATA_DIR, NS_DATA_DIR
from utils.converting_raw_data import transform_raw_data

np.random.seed(RANDOM_STATE)


def load_data(data_path, filename):
    with open(osp.join(data_path, f'{filename}.pckl'), 'rb') as handle:
        data = pickle.load(handle)
    return data


def save_data(file, data_path, filename):
    with open(osp.join(data_path, f'{filename}.pckl'), 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_train_valid_test_data(data: dict,
                                 labels: list,
                                 train_size: float = 0.75,
                                 valid_size: float = 0.1,
                                 test_size: float = 0.15):
    """
        Given data and labels, split it into train,
        valid and test data and labels and then save it.
    """

    indices = np.arange(len(data["scatter"]))
    np.random.shuffle(indices)

    # shuffle data and labels
    for feature in data.keys():
        data[feature] = np.array(data[feature])[indices]
    labels = np.array(labels)[indices]

    train_indices = (0, int(train_size * len(data["scatter"])))
    valid_indices = (train_indices[-1], train_indices[-1] + int(valid_size * len(data["scatter"])))
    test_indices = (valid_indices[-1], valid_indices[-1] + int(test_size * len(data["scatter"])))

    train_data = []
    valid_data = []
    test_data = []

    for feature in data:
        train_data.extend([np.array(data[feature])[train_indices[0]: train_indices[1]]])
        valid_data.extend([np.array(data[feature])[valid_indices[0]: valid_indices[1]]])
        test_data.extend([np.array(data[feature])[test_indices[0]: test_indices[1]]])

    train_labels = np.array(labels)[train_indices[0]: train_indices[1]]
    valid_labels = np.array(labels)[valid_indices[0]: valid_indices[1]]
    test_labels = np.array(labels)[test_indices[0]: test_indices[1]]

    data_to_save = [train_data, valid_data, test_data]
    labels_to_save = [train_labels, valid_labels, test_labels]

    return data_to_save, labels_to_save


def flatten_data(data):
    new_train_data = np.zeros((len(data["scatter"]), 2629))
    for vector_index in range(len(data["scatter"])):
        new_vector = []
        for feature_index, vector in enumerate(data):
            if feature_index == 1:
                new_vector.extend([vector[vector_index]])
            else:
                new_vector.extend(np.array(vector[vector_index]).flatten())
        new_train_data[vector_index] = new_vector
    return new_train_data


def find_statistical_components(data):
    data = np.array(data).flatten()

    min_value = min(data)
    max_value = max(data)

    mean_value = np.mean(data)
    std_value = np.std(data)

    return {'min': min_value, 'max': max_value, 'mean': mean_value, 'std': std_value}


def normalize_data(data, mean_value, std_value):
    data -= mean_value
    data /= std_value

    return data


def standardize_data(data, min_value, max_value):
    data -= min_value
    data /= max_value - min_value

    return data


def split_and_save_data(raw_data_path,
                        output_data_path,
                        data_normalization=True,
                        data_standardization=True):
    print(f'Started transforming raw data at {datetime.now().time()}')
    feature_names, files, filenames = transform_raw_data(raw_data_path)

    print(f'Started creating train/test data {datetime.now().time()}')
    data_to_save, labels_to_save = create_train_valid_test_data(data=files["data"],
                                                                labels=files["labels"])
    dirs_to_save = ['train', 'valid', 'test']
    for feature_index, feature in enumerate(feature_names):
        print(f'Current feature is {feature} {datetime.now().time()}')

        for dir_index, data_type in enumerate(dirs_to_save):
            print(f'Current data type is {data_type} {datetime.now().time()}')
            data_path = osp.join(output_data_path, data_type)

            if not osp.exists(data_path):
                os.makedirs(data_path)

            if data_type == "train":
                stat_comp = find_statistical_components(data_to_save[dir_index][feature_index])
                save_data(stat_comp, data_path, filename=f'{feature}_stat_comp')

            save_data(file=data_to_save[dir_index][feature_index],
                      data_path=data_path,
                      filename=feature)

            if data_standardization:
                standardized_path = osp.join(data_path, 'standardized_data')
                if not osp.exists(standardized_path):
                    os.makedirs(standardized_path)

                save_data(file=standardize_data(data_to_save[dir_index][feature_index],
                                                stat_comp["min"],
                                                stat_comp["max"]),
                          data_path=standardized_path,
                          filename=feature)

            if data_normalization:
                normalize_path = osp.join(data_path, 'normalized_data')
                if not osp.exists(normalize_path):
                    os.makedirs(normalize_path)

                save_data(file=normalize_data(data_to_save[dir_index][feature_index],
                                              stat_comp["mean"],
                                              stat_comp["std"]),
                          data_path=normalize_path,
                          filename=feature)

        save_data(file=labels_to_save[dir_index],
                  data_path=data_path,
                  filename="target")


if __name__ == '__main__':
    split_data = True
    standardize = True
    normalize = True
    raw_data_dir = NS_RAW_DATA_DIR
    output_data_dir = NS_DATA_DIR

    data, labels, label_to_index, feature_names = transform_raw_data(NS_RAW_DATA_DIR)

    if split_data:
        [train_data, valid_data, test_data], [train_labels, valid_labels, test_labels] = create_train_valid_test_data(
            data=data,
            labels=labels)

        for feature_index, feature in enumerate(train_data):
            stat_comp = find_statistical_components(feature)
            save_data(file=stat_comp,
                      data_path=output_data_dir,
                      filename=f'stat_comp{feature_names[feature_index]}')

            save_data(file=labels[feature_index],
                      data_path=output_data_dir,
                      filename="labels")

            for data in [train_data, valid_data, test_data]:
                if normalize:
                    save_data(file=normalize_data(data[feature_index],
                                                  stat_comp["mean"],
                                                  stat_comp["std"]),
                              data_path=osp.join(output_data_dir, 'normalized'),
                              filename=feature_names[feature_index])
                if standardize:
                    save_data(file=standardize_data(data[feature_index],
                                                    stat_comp["min"],
                                                    stat_comp["max"]),
                              data_path=osp.join(output_data_dir, 'standardized'),
                              filename=feature)

    print()
    # split_and_save_data(NS_RAW_DATA_DIR, NS_DATA_DIR)
