from typing import List

from keras.engine import Layer
from keras.layers import Activation, Add, GlobalAvgPool2D

from source.models.convolutional_layers.cnn_layers import create_cnn_layer


def conv1(input_layer: "Layer",
          filters: int) -> "Layer":
    """
    :param input_layer:
    :param filters:
    :return:
    """

    layer = create_cnn_layer(input_layer, filters, kernel_size=1)
    return layer


def conv1_downsample(input_layer: "Layer",
                     filters: int) -> "Layer":
    """
    :param input_layer:
    :param filters:
    :return:
    """

    layer = create_cnn_layer(input_layer, filters, kernel_size=1)
    return layer


def conv3(input_layer: "Layer",
          filters: int):
    """
    :param input_layer:
    :param filters:
    :return:
    """

    layer = create_cnn_layer(input_layer, filters)
    return layer


def conv3_downsample(input_layer: "Layer",
                     filters: int):
    """
    :param input_layer:
    :param filters:
    :return:
    """

    layer = create_cnn_layer(input_layer, filters)
    return layer


def resnet_block_wo_bottlneck(input_layer: "Layer",
                              filters: int,
                              downsample: bool = False) -> "Layer":
    """
    :param input_layer:
    :param filters:
    :param downsample:
    :return:
    """

    if downsample:
        layer = conv3_downsample(input_layer, filters)
    else:
        layer = conv3(input_layer, filters)

    layer = conv3(layer, filters)
    if downsample:
        input_layer = conv1_downsample(input_layer, filters)

    layer = Add()([layer, input_layer])
    layer = Activation('relu')(layer)

    return layer


def resnet_block_w_bottlneck(input_layer: "Layer",
                             filters: int,
                             downsample: bool = False,
                             change_channels: bool = False) -> "Layer":
    """
    :param input_layer:
    :param filters:
    :param downsample:
    :param change_channels:
    :return:
    """

    if downsample:
        layer = conv1_downsample(input_layer, int(filters / 4))
    else:
        layer = conv1(input_layer, int(filters / 4))

    layer = conv3(layer, int(filters / 4))
    layer = conv1(layer, filters)

    if downsample:
        input_layer = conv1_downsample(input_layer, filters)
    elif change_channels:
        input_layer = conv1(input_layer, filters)

    result = Add()([layer, input_layer])

    return result


def _pre_res_blocks(input_layer: "Layer") -> "Layer":
    layer = create_cnn_layer(input_layer, 64, 3)
    return layer


# def _post_res_blocks(self, in_tensor):
#     pool = layers.GlobalAvgPool2D()(in_tensor)
#     # preds = layers.Dense(self.num_classes, activation='softmax')(pool)
#     return preds


def convx_wo_bottleneck(input_layer: "Layer",
                        filters: int,
                        n_times: int,
                        downsample_1: bool = False) -> "Layer":
    """
    :param input_layer:
    :param filters:
    :param n_times:
    :param downsample_1:
    :return:
    """

    layer = input_layer
    for i in range(n_times):
        if i == 0:
            layer = resnet_block_wo_bottlneck(layer, filters, downsample_1)
        else:
            layer = resnet_block_wo_bottlneck(layer, filters)

    return layer


def convx_w_bottleneck(input_layer: "Layer",
                       filters: int,
                       n_times: int,
                       downsample_1: bool = False) -> "Layer":
    """
    :param input_layer:
    :param filters:
    :param n_times:
    :param downsample_1:
    :return:
    """

    layer = input_layer

    for i in range(n_times):
        if i == 0:
            layer = resnet_block_w_bottlneck(layer, filters, downsample_1, not downsample_1)
        else:
            layer = resnet_block_w_bottlneck(layer, filters)

    return layer


def create_residual(in_layer: "Layer",
                    convx: List[int],
                    n_convx: List[int]):
    """
    :param in_layer:
    :param convx:
    :param n_convx:
    :return:
    """

    convx_fn = convx_w_bottleneck
    downsampled = _pre_res_blocks(in_layer)

    layer = convx_fn(downsampled, convx[0], n_convx[0])
    layer = convx_fn(layer, convx[1], n_convx[1])
    layer = convx_fn(layer, convx[2], n_convx[2])

    layer = GlobalAvgPool2D()(layer)
    return layer


