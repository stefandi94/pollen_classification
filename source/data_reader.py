import pickle
from typing import List, Tuple

import numpy as np
import os.path as osp

from settings import TRAINING_DIR, VALID_DIR, TEST_DIR


def create_3_channels(array: np.ndarray) -> np.ndarray:
    """
    Given 3d array, same channel 3 times
    :param array:
    :return:
    """

    new_array = []
    for x in array:
        new_array.append(np.concatenate((x, x, x), axis=2))
    new_array = np.array(new_array)
    return new_array


def create_3d_array(array: np.ndarray) -> np.ndarray:
    """Given 2d array reshape it to 3d
    :param array:
    :return: array
    """
    array = np.reshape(array, (array.shape[0], array.shape[1], array.shape[2], 1))
    return array


def read_data(name: str, tip: str, normalized: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param name: Name of the type of feature to load
    :param tip: Train, valid or test data
    :param normalized:
    :return: Tuple with data as first element and classes as second
    """

    if tip == 'train':
        dire = TRAINING_DIR
    elif tip == 'valid':
        dire = VALID_DIR
    elif tip == 'test':
        dire = TEST_DIR

    with open(osp.join(dire, 'target.pckl'), 'rb') as fp:
        y = pickle.load(fp)

    if normalized:
        name = 'normalized_' + name

    with open(osp.join(dire, f'{name}.pckl'), 'rb') as fp:
        X = pickle.load(fp)

    return X, np.array(y)


def load_all_data(tip: str, normalized: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    :param tip: Which data to load - train, valid or test
    :param normalized: Boolean to normalize data with keras normalize function
    :return:
    """

    X_size, _ = read_data('size', tip, normalized)
    X_life_1, _ = read_data('life_1', tip, normalized)
    X_life_2_cut, _ = read_data('life_2_cut', tip, normalized)
    X_image, y_image = read_data('image', tip, normalized)

    return [X_image, X_life_1, X_size, X_life_2_cut], y_image
