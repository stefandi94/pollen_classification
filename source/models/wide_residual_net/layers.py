from typing import Tuple

from keras.engine import Layer
from keras.layers import BatchNormalization, Activation, Add, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2

from source.models.convolutional_layers.cnn_layers import create_cnn_layer

weight_decay = 0.0005


def initial_conv(input_layer: "Layer") -> "Layer":
    """
    :return:
    """

    layer = create_cnn_layer(input_layer=input_layer,
                             num_filter=16,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay))
    return layer


def expand_conv(input_layer: "Layer",
                base: int or Tuple[int, int],
                k: int,
                strides: int or Tuple[int, int] = (1, 1)) -> "Layer":
    """
    :param input_layer:
    :param base:
    :param k:
    :param strides:
    :return:
    """

    layer = create_cnn_layer(input_layer,
                             base * k,
                             strides=strides,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay))

    layer = create_cnn_layer(layer,
                             base * k,
                             strides=strides,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay),
                             batch_normalization=False,
                             activation=False)

    skip = create_cnn_layer(input_layer,
                            base * k,
                            kernel_size=(1, 1),
                            strides=strides,
                            kernel_init='he_normal',
                            kernel_regularizer=l2(weight_decay))

    layer_after_addition = Add()([layer, skip])

    return layer_after_addition


def conv1_block(input_layer: "Layer",
                k: int = 1,
                dropout: float = 0.0) -> "Layer":
    """
    :param input_layer:
    :param k:
    :param dropout:
    :return:
    """

    layer = BatchNormalization(momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform')(input_layer)

    layer = Activation('relu')(layer)

    layer = create_cnn_layer(layer,
                             16 * k,
                             dropout=dropout,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay))

    x = create_cnn_layer(layer,
                         16 * k,
                         kernel_init='he_normal',
                         kernel_regularizer=l2(weight_decay))

    layer_after_addition = Add()([input_layer, x])
    return layer_after_addition


def conv2_block(input_layer: "Layer",
                k: int = 1,
                dropout: float = 0.0) -> "Layer":
    """
    :param input_layer:
    :param k:
    :param dropout:
    :return:
    """

    layer = BatchNormalization(momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform')(input_layer)

    layer = Activation('relu')(layer)

    layer = create_cnn_layer(layer,
                             32 * k,
                             dropout=dropout,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay))

    layer = create_cnn_layer(layer,
                             32 * k,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay))

    layer_after_addition = Add()([input_layer, layer])
    return layer_after_addition


def conv3_block(input_layer: "Layer",
                k: int = 1,
                dropout: float = 0.0) -> "Layer":
    """
    :param input_layer:
    :param k:
    :param dropout:
    :return:
    """

    layer = BatchNormalization(momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform')(input_layer)

    layer = Activation('relu')(layer)
    layer = create_cnn_layer(layer,
                             64 * k,
                             dropout=dropout,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay))

    layer = create_cnn_layer(layer,
                             64 * k,
                             kernel_init='he_normal',
                             kernel_regularizer=l2(weight_decay))

    layer_after_addition = Add()([input_layer, layer])
    return layer_after_addition


def create_wide_residual_network(k: int,
                                 N: int,
                                 input_shape: "Layer",
                                 dropout: float = 0.0) -> "Layer":
    """
    Creates a Wide Residual Network with specified parameters
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.

    :param input_shape: Input Keras object
    :param dropout: Adds dropout if value is greater than 0.0
    :return:
    """

    layer = initial_conv(input_shape)
    nb_conv = 4

    layer = expand_conv(layer, 16, k)
    nb_conv += 2

    for i in range(N - 1):
        layer = conv1_block(layer, k, dropout)
        nb_conv += 2

    layer = BatchNormalization(momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform')(layer)
    layer = Activation('relu')(layer)

    layer = expand_conv(layer, 32, k, strides=(1, 1))
    nb_conv += 2

    for i in range(N - 1):
        layer = conv2_block(layer, k, dropout)
        nb_conv += 2

    layer = BatchNormalization(momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform')(layer)
    layer = Activation('relu')(layer)

    layer = expand_conv(layer, 64, k, strides=(1, 1))
    nb_conv += 2

    for i in range(N - 1):
        layer = conv3_block(layer, k, dropout)
        nb_conv += 2

    layer = BatchNormalization(momentum=0.1,
                               epsilon=1e-5,
                               gamma_initializer='uniform')(layer)
    layer = Activation('relu')(layer)

    layer = GlobalAveragePooling2D()(layer)
    layer = Dropout(dropout)(layer)

    return layer
