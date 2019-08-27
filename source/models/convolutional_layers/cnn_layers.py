from functools import partial
from typing import List, Tuple

import keras
from keras.engine import Layer
from keras.initializers import Initializer
from keras.layers import Conv2D, BatchNormalization, GlobalAvgPool2D, LeakyReLU, SpatialDropout2D, \
    Conv1D, Activation, Dropout
from keras.regularizers import Regularizer

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)


def create_cnn_layer(input_layer: "Layer",
                     num_filter: int,
                     kernel_size: int or Tuple[int, int] = (3, 3),
                     dropout: float = 0.0,
                     batch_normalization: bool = False,
                     kernel_init: str or "Initializer" = kernel_init,
                     bias_init: str or "Initializer" = bias_init,
                     strides: int or Tuple[int, int] = (1, 1),
                     kernel_regularizer: Regularizer = None,
                     activation: bool = True,
                     one_d: bool = False) -> "Layer":
    """
    Given input layer and number of filters, do 2D convolution
    :param input_layer: Input layer
    :param num_filter: Number of feature maps
    :param batch_normalization
    :param dropout:
    :param kernel_init:
    :param bias_init:
    :param kernel_size:
    :param strides
    :param kernel_regularizer
    :param activation
    :param one_d: whether to do 1D or 2D convolution
    :return: Layer
    """
    if one_d:
        conv = Conv1D
    else:
        conv = Conv2D

    layer = conv(num_filter,
                 strides=strides,
                 kernel_size=kernel_size,
                 padding='same',
                 kernel_initializer=kernel_init,
                 bias_initializer=bias_init,
                 kernel_regularizer=kernel_regularizer)(input_layer)

    if batch_normalization:
        layer = BatchNormalization()(layer)
    if activation:
        layer = Activation('relu')(layer)

    layer = Dropout(dropout)(layer)

    return layer


def create_cnn_network(input_layer: "Layer",
                       num_of_filters: List[int],
                       dropout: float = 0,
                       batch_normalization: bool = False,
                       kernels: List[int] or List[Tuple] = None,
                       one_d: bool = False) -> "Layer":
    """
    Given input layer and number of filters, creates network
    :param input_layer:
    :param num_of_filters:
    :param kernels:
    :param dropout
    :param batch_normalization
    :param one_d: Conv1D or Conv2D
    :return:
    """
    if kernels is None:
        kernels = [(2, 2)] * len(num_of_filters)

    layer = input_layer
    for index, filter in enumerate(num_of_filters):
        layer = create_cnn_layer(input_layer=layer,
                                 num_filter=filter,
                                 kernel_size=kernels[index],
                                 dropout=dropout,
                                 batch_normalization=batch_normalization,
                                 one_d=one_d)
    layer = GlobalAvgPool2D()(layer)
    return layer
