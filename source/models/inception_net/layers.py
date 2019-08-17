from typing import Tuple

from keras.engine import Layer
from keras.layers import Conv2D, concatenate, GlobalAveragePooling2D

from source.models.convolutional_layers.cnn_layers import create_cnn_layer


def inception_module(input_layer: "Layer",
                     filters_1x1: int,
                     filters_3x3_reduce: int,
                     filters_3x3: int,
                     filters_5x5_reduce: int,
                     filters_5x5: int):
    """
    :param input_layer:
    :param filters_1x1:
    :param filters_3x3_reduce:
    :param filters_3x3:
    :param filters_5x5_reduce:
    :param filters_5x5:
    :return:
    """

    conv_1x1 = create_cnn_layer(input_layer, filters_1x1, kernel_size=(1, 1))

    conv_3x3 = create_cnn_layer(input_layer, filters_3x3_reduce, kernel_size=(1, 1))
    conv_3x3 = create_cnn_layer(conv_3x3, filters_3x3, kernel_size=(3, 3))

    conv_5x5 = create_cnn_layer(input_layer, filters_5x5_reduce, kernel_size=(1, 1))
    conv_5x5 = create_cnn_layer(conv_5x5, filters_5x5, kernel_size=(5, 5))

    output = concatenate([conv_1x1, conv_3x3, conv_5x5], axis=3)
    return output


def create_inception_model(input_layer: "Layer") -> Tuple["Layer", "Layer", "Layer"]:
    """
    :param input_layer:
    :return:
    """

    layer = create_cnn_layer(input_layer, 32, strides=(1, 2))
    layer = create_cnn_layer(layer, 64, kernel_size=(1, 1))
    layer = create_cnn_layer(layer, 64)

    layer = inception_module(layer,
                             filters_1x1=32,
                             filters_3x3_reduce=64,
                             filters_3x3=128,
                             filters_5x5_reduce=32,
                             filters_5x5=64)

    layer = inception_module(layer,
                             filters_1x1=64,
                             filters_3x3_reduce=64,
                             filters_3x3=128,
                             filters_5x5_reduce=16,
                             filters_5x5=32)

    auxilliary_output_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(layer)
    auxilliary_output_1 = GlobalAveragePooling2D()(auxilliary_output_1)

    layer = inception_module(layer,
                             filters_1x1=64,
                             filters_3x3_reduce=64,
                             filters_3x3=32,
                             filters_5x5_reduce=32,
                             filters_5x5=64)

    layer = inception_module(layer,
                             filters_1x1=64,
                             filters_3x3_reduce=64,
                             filters_3x3=128,
                             filters_5x5_reduce=32,
                             filters_5x5=64)

    auxilliary_output_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(layer)
    auxilliary_output_2 = GlobalAveragePooling2D()(auxilliary_output_2)

    layer = inception_module(layer,
                             filters_1x1=64,
                             filters_3x3_reduce=64,
                             filters_3x3=128,
                             filters_5x5_reduce=32,
                             filters_5x5=32)

    layer = inception_module(layer,
                             filters_1x1=64,
                             filters_3x3_reduce=128,
                             filters_3x3=64,
                             filters_5x5_reduce=64,
                             filters_5x5=128)

    layer = GlobalAveragePooling2D()(layer)
    return layer, auxilliary_output_2, auxilliary_output_1
