from typing import Tuple

from keras.engine import Layer
from keras.layers import GlobalAveragePooling2D, Reshape, Dropout, DepthwiseConv2D, BatchNormalization, Conv2D, add, \
    ReLU, K

relu6 = ReLU(max_value=6)


def _conv_block(inputs: "Layer",
                filters: int,
                kernel: int or Tuple[int, int],
                strides: int or Tuple[int, int]) -> "Layer":
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    layer = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    layer = BatchNormalization()(layer)
    layer = relu6(layer)
    return layer


def _bottleneck(inputs: "Layer",
                filters: int,
                kernel: int or Tuple[int, int],
                expansion_factor: int,
                strides: int or Tuple[int, int],
                residuals: bool = False) -> "Layer":
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    tchannel = K.int_shape(inputs)[1] * expansion_factor

    layer = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    layer = DepthwiseConv2D(kernel, strides=(strides, strides), depth_multiplier=1, padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = relu6(layer)

    layer = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(layer)
    layer = BatchNormalization()(layer)

    if residuals:
        layer = add([layer, inputs])
    return layer


def _inverted_residual_block(inputs: "Layer",
                             filters: int,
                             kernel: int or Tuple[int, int],
                             t: int,
                             strides: int or Tuple[int, int],
                             n: int) -> "Layer":
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    layer = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        layer = _bottleneck(layer, filters, kernel, t, 1, True)

    return layer


def mobile_net_v2(inputs):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    layer = _conv_block(inputs, 32, (3, 3), strides=(1, 1))

    for i in [16, 24, 32, 64, 96, 160, 320]:
        layer = _inverted_residual_block(layer, i, (3, 3), t=1, strides=1, n=1)

    layer = _conv_block(layer, 300, (1, 1), strides=(1, 1))
    layer = GlobalAveragePooling2D()(layer)
    layer = Reshape((1, 1, 300))(layer)
    layer = Dropout(0.3, name='Dropout')(layer)

    return layer
