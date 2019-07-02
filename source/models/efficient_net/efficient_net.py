# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

from keras import Input, Model
from keras.layers import Dropout, concatenate, Dense

from source.base_dl_model import BaseDLModel
from source.models.efficient_net.layers import _get_model_by_name

BaseDLModel
_get_model_by_name

__all__ = ['EfficientNet', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
           'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']


class EfficientNet(BaseDLModel):

    def __init__(self, **parameters):

        super().__init__(**parameters)

    def build_model(self) -> "None":
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [Dropout(0.25)(layer) for layer in inputs]
        layers = [EfficientNetB0(inputs=layer) for layer in layers]
        layers = [Dropout(0.25)(layer) for layer in layers]
        layer = concatenate([layer for layer in layers])
        layer = Dropout(0.5)(layer)

        output = Dense(self.num_classes, activation='relu')(layer)
        model = Model(inputs, output)

        self.model = model


def EfficientNetB0(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b0', include_top=include_top, inputs=inputs, weights=weights)


def EfficientNetB1(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b1', include_top=include_top, inputs=inputs, weights=weights)


def EfficientNetB2(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b2', include_top=include_top, inputs=inputs, weights=weights)


def EfficientNetB3(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b3', include_top=include_top, inputs=inputs, weights=weights)


def EfficientNetB4(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b4', include_top=include_top, inputs=inputs, weights=weights)


def EfficientNetB5(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b5', include_top=include_top, inputs=inputs, weights=weights)


def EfficientNetB6(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b6', include_top=include_top, inputs=inputs, weights=weights)


def EfficientNetB7(include_top=False, inputs=None, weights=None):
    return _get_model_by_name('efficient_net-b7', include_top=include_top, inputs=inputs, weights=weights)


EfficientNetB0.__doc__ = _get_model_by_name.__doc__
EfficientNetB1.__doc__ = _get_model_by_name.__doc__
EfficientNetB2.__doc__ = _get_model_by_name.__doc__
EfficientNetB3.__doc__ = _get_model_by_name.__doc__
EfficientNetB4.__doc__ = _get_model_by_name.__doc__
EfficientNetB5.__doc__ = _get_model_by_name.__doc__
EfficientNetB6.__doc__ = _get_model_by_name.__doc__
EfficientNetB7.__doc__ = _get_model_by_name.__doc__
