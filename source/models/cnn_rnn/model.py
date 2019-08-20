from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dense, LSTM, Reshape, Dropout
from keras.regularizers import l1_l2

from settings import KERNEL_REGULARIZER, ACTIVITY_REGULARIZER, BIAS_REGULARIZER
from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network
from source.models.dense_layers.dense_layers import create_dense_network


class CNNRNN(BaseDLModel):
    num_of_neurons = [[100],
                      [100],
                      [50]]
    convolution_filters = [64, 128, 128]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs_1 = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers_1 = [Flatten()(layer) for layer in inputs_1]
        layers_1 = [create_cnn_network(layer, [100], one_d=True) for layer in layers_1]
        layer_1 = concatenate([layer for layer in layers_1])
        layer_1 = Reshape((int(layer_1.shape[1]), 1))(layer_1)
        layer_1 = LSTM(128, return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l1_l2(KERNEL_REGULARIZER),
                       bias_regularizer=l1_l2(BIAS_REGULARIZER), activity_regularizer=l1_l2(ACTIVITY_REGULARIZER))(
            layer_1)

        layers_2 = [create_cnn_network(layer, self.convolution_filters) for layer in inputs_1]
        layer_2 = concatenate([layer for layer in layers_2])

        layer = concatenate([layer_1, layer_2])
        flatten = Dropout(0.5)(layer)
        output = Dense(self.num_classes, activation='softmax')(flatten)
        model = Model(inputs_1, output)
        self.model = model
