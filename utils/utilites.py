from collections import Generator
from typing import List

import keras
import numpy as np
from keras.optimizers import Adam, SGD, Optimizer
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight


def create_optimizer(opt_name: str,
                     lr: float,
                     weight_decay: float = 0) -> "Optimizer":
    """
    :param opt_name: optimizer name
    :param lr: learning rate
    :param weight_decay:
    :return: optimizer function
    """

    if opt_name == 'adam':
        optimizer = Adam(lr, decay=weight_decay)
    elif opt_name == 'sgd':
        optimizer = SGD(lr, decay=weight_decay)
    else:
        raise ValueError('Not know optimizer, choose between adam or sgd')

    return optimizer


def smooth_labels(y: np.ndarray,
                  smooth_factor: float) -> np.ndarray:
    """Convert a matrix of one-hot row-vector labels into smoothed versions.
    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
    # Returns
        A matrix of smoothed labels.
    """

    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def multiple_generator(X: np.ndarray or List[np.ndarray],
                       y: np.ndarray,
                       batch_size: int) -> "Generator":
    """
    Given data, labels, generator and batch size create generator for all training data, yielding one at the time
    :param X: training data, containing four dimensions, for each feature
    :param y: labels
    :param batch_size:
    :return: generator, which will yield list with all features
    """

    list_of_generators = []
    for x in X:
        generator = create_generator()
        generator.fit(x, augment=True)
        generator = generator.flow(x, y, batch_size=batch_size)
        list_of_generators.append(generator)

    while True:

        generator_data = []
        generator_class = None

        for generator in list_of_generators:
            x, y = generator.next()

            generator_data.append(x)
            if generator_class is None:
                generator_class = y

        yield generator_data, generator_class  # Yield both images and their mutual label


def join_life_2_cut(life_2: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """
    :param life_2: life_2 feature
    :param cut: cut feature
    :return: Joined matrix (x, 5, 1, 1) dimensions, where first 4 dimensions are life_2 and last one is cut.
    """

    X_joined = np.empty(shape=(len(cut), 5, 1, 1))
    X_joined[:, :4, 0, 0] = life_2
    X_joined[:, 4, 0, 0] = cut

    return X_joined


def calculate_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Given class labels of training sample, calculates frequency of each class and return inverted proportion, to
    balance dataset.
    :param y_train: training labels
    :return: weight class for each label
    """

    weight_class = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return weight_class


def create_generator() -> "ImageDataGenerator":
    """
    Create ImageDataGenerator which will output data will have Normal(0, 1) distribution
    :return:
    """

    datagen = ImageDataGenerator(featurewise_std_normalization=True,
                                 featurewise_center=True)
    return datagen


def normalize(X: np.ndarray) -> np.ndarray:
    X = keras.utils.normalize(X, axis=-1)
    return X


def count_values(array):
    """
    Given array return dictionary with class numbers and number of instances of that class
    :param array:
    :return:
    """
    unique, counts = np.unique(array, return_counts=True)
    return dict(zip(unique, counts))


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
