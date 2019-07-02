import math
from typing import List, Any

import keras.backend as K
import keras.layers as KL
import numpy as np
import tensorflow as tf
from keras.initializers import Initializer
from keras.layers import GlobalAveragePooling2D
from keras.utils import get_file

from .special_layers import Swish, DropConnect
from .params import get_model_params, IMAGENET_WEIGHTS


def conv_kernel_initializer(shape: List[int],
                            dtype: type = K.floatx()) -> Initializer:
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable

    Returns:
      an initialization for the variable
    """
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape: List[int],
                             dtype: type = K.floatx()) -> Initializer:
    """Initialization for dense_layers kernels.

    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable

    Returns:
      an initialization for the variable
    """

    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters: int,
                  global_params: Any) -> int:
    """Round number of filters based on depth multiplier."""

    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    # print('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def SEBlock(block_args, global_params):
    num_reduced_filters = max(
        1, int(block_args.input_filters * block_args.se_ratio))
    filters = block_args.input_filters * block_args.expand_ratio
    if global_params.data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = KL.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = KL.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=True
        )(x)
        x = Swish()(x)
        # Excite
        x = KL.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=True
        )(x)
        x = KL.Activation('sigmoid')(x)
        out = KL.Multiply()([x, inputs])
        return out

    return block


def MBConvBlock(block_args, global_params):
    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon

    if global_params.data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (block_args.se_ratio is not None) and (
            block_args.se_ratio > 0) and (block_args.se_ratio <= 1)

    filters = block_args.input_filters * block_args.expand_ratio
    kernel_size = block_args.kernel_size

    def block(inputs):

        if block_args.expand_ratio != 1:
            x = KL.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False
            )(inputs)
            x = KL.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon
            )(x)
            x = Swish()(x)
        else:
            x = inputs

        x = KL.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False
        )(x)
        x = KL.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(block_args, global_params)(x)

        # output phase

        x = KL.Conv2D(
            block_args.output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False
        )(x)
        x = KL.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon
        )(x)

        if block_args.id_skip:
            if all(
                    s == 1 for s in block_args.strides
            ) and block_args.input_filters == block_args.output_filters:
                # only apply drop_connect if skip presents.
                if global_params.drop_connect_rate:
                    x = DropConnect(global_params.drop_connect_rate)(x)
                x = KL.Add()([x, inputs])
        return x

    return block


def EfficientNet(inputs, block_args_list, global_params):
    batch_norm_momentum = global_params.batch_norm_momentum
    batch_norm_epsilon = global_params.batch_norm_epsilon
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Stem part
    x = inputs
    x = KL.Conv2D(
        filters=round_filters(8, global_params),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False
    )(x)
    x = KL.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon
    )(x)
    x = Swish()(x)

    # Blocks part
    for block_args in block_args_list:
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters, global_params),
            output_filters=round_filters(block_args.output_filters, global_params),
            num_repeat=round_repeats(block_args.num_repeat, global_params)
        )

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args, global_params)(x)

        if block_args.num_repeat > 1:
            block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args, global_params)(x)

    # Head part
    x = KL.Conv2D(
        filters=round_filters(500, global_params),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False
    )(x)
    x = KL.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon
    )(x)
    x = Swish()(x)
    x = GlobalAveragePooling2D()(x)
    return x


def _get_model_by_name(model_name, inputs=None, include_top=True, weights=None, classes=50):
    """Re-Implementation of EfficientNet for Keras

    Reference:
        https://arxiv.org/abs/1807.11626

    Args:
        inputs: optional, if ``None`` default_input_shape is used
            EfficientNetB0 - (224, 224, 3)
            EfficientNetB1 - (240, 240, 3)
            EfficientNetB2 - (260, 260, 3)
            EfficientNetB3 - (300, 300, 3)
            EfficientNetB4 - (380, 380, 3)
            EfficientNetB5 - (456, 456, 3)
            EfficientNetB6 - (528, 528, 3)
            EfficientNetB7 - (600, 600, 3)
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet).
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    """
    if weights not in {None, 'imagenet'}:
        raise ValueError('Parameter `weights` should be one of [None, "imagenet"]')

    if weights == 'imagenet' and model_name not in IMAGENET_WEIGHTS:
        raise ValueError('There are not pretrained weights for {} model.'.format(model_name))

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` and `include_top`'
                         ' `classes` should be 1000')

    block_agrs_list, global_params, default_input_shape = get_model_params(
        model_name, override_params={'num_classes': classes}
    )

    # if inputs is None:
    #     inputs = (default_input_shape, default_input_shape, 3)

    model = EfficientNet(inputs, block_agrs_list, global_params)
    # model.name = model_name

    if weights:
        if not include_top:
            weights_name = model_name + '-notop'
        else:
            weights_name = model_name
        weights = IMAGENET_WEIGHTS[weights_name]
        weights_path = get_file(
            weights['name'],
            weights['url'],
            cache_subdir='lstm_net',
            md5_hash=weights['md5'],
        )
        model.load_weights(weights_path)

    return model
