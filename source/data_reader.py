import pickle
from typing import List, Tuple

import numpy as np
import os.path as osp

from settings import NS_TRAIN_DIR, NS_VALID_DIR, NS_TEST_DIR
from utils.split_data import load_data


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
    array = np.reshape(array, (array.shape[0], array.shape[1], 1))
    return array


def create_4d_array(array: np.ndarray) -> np.ndarray:
    array = np.reshape(array, (array.shape[0], array.shape[1], array.shape[2], 1))
    return array


def read_data(name: str, tip: str, load_labels: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param name: Name of the type of feature to load
    :param tip: Train, valid or test data
    :return: Tuple with data as first element and classes as second
    """

    if tip == 'train':
        dire = NS_TRAIN_DIR
    elif tip == 'valid':
        dire = NS_VALID_DIR
    elif tip == 'test':
        dire = NS_TEST_DIR

    with open(osp.join(dire, 'target.pckl'), 'rb') as fp:
        y = pickle.load(fp)

    with open(osp.join(dire, f'{name}.pckl'), 'rb') as fp:
        X = pickle.load(fp)

    if load_labels:
        return X, np.array(y)
    else:
        return X


def load_all_data(data_path: str) -> Tuple[List[np.ndarray], np.ndarray]:

    X_scatter = load_data(data_path, 'scatter')
    X_size = load_data(data_path, 'size')
    X_life_1 = load_data(data_path, 'life_1')
    X_spectrum = load_data(data_path, 'spectrum')
    X_life_2 = load_data(data_path, 'life_2')
    y = load_data(data_path, 'labels')

    return [X_scatter, X_size, X_life_1, X_spectrum, X_life_2], y
