from typing import List

from keras.engine import Layer
from keras.layers import Dense, BatchNormalization, Activation, Dropout, LeakyReLU
from keras.regularizers import l1_l2

from settings import KERNEL_REGULARIZER, BIAS_REGULARIZER, ACTIVITY_REGULARIZER


def create_dense_network(input_layer: "Layer",
                         batch_normalization: bool = True,
                         dropout: float = 0.3,
                         num_of_neurons: List[int] = [50, 50, 20]) -> "Layer":
    """
    :param input_layer: Input layer
    :param dropout: Percentage of neurons to drop
    :param batch_normalization: Boolean, if to normalize batch
    :param num_of_neurons: List of with number of neurons in neural network
    :return: Layer
    """

    layer = input_layer

    for neuron in num_of_neurons:
        layer = dense_layer(layer, neuron, dropout, batch_normalization)

    return layer


def dense_layer(input_layer: "Layer",
                num_of_neurons: int,
                dropout: float,
                batch_normalization: bool = True,
                activation: bool = True) -> "Layer":
    """
    :param input_layer:
    :param num_of_neurons:
    :param dropout:
    :param batch_normalization:
    :param activation:
    :return:
    """

    layer = Dense(num_of_neurons)(input_layer)

    if batch_normalization:
        layer = BatchNormalization()(layer)

    if activation:
        layer = LeakyReLU()(layer)

    layer = Dropout(dropout)(layer)
    return layer
