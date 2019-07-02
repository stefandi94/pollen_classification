from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dense, LSTM, Reshape, Dropout, BatchNormalization, Activation, LeakyReLU
from keras.regularizers import l1_l2

from settings import BIAS_REGULARIZER, ACTIVITY_REGULARIZER, KERNEL_REGULARIZER
from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network
from source.models.dense_layers.dense_layers import create_dense_network


class ANNCNNRNN(BaseDLModel):
    num_of_neurons = [[100],
                      [100],
                      [50]]
    convolution_filters = [64, 128, 256]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs_1 = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers_1 = [Flatten()(layer) for layer in inputs_1]
        layers_1 = [create_dense_network(layer, num_of_neurons=[100, 100]) for layer in layers_1]
        layer_1 = concatenate([layer for layer in layers_1])
        layer_1 = Reshape((int(layer_1.shape[1]), 1))(layer_1)
        layer_1 = LSTM(256, return_sequences=False, recurrent_dropout=0.25, dropout=0.25)(layer_1)

        inputs_2 = [Input(input_shape) for input_shape in self.cnn_shapes.values()]
        layers_2 = [create_dense_network(layer, num_of_neurons=[100, 100]) for layer in inputs_2]
        layers_2 = [create_cnn_network(layer, self.convolution_filters) for layer in layers_2]
        layer_2 = concatenate([layer for layer in layers_2])

        layer = concatenate([layer_1, layer_2])
        layer = BatchNormalization()(layer)
        layer = LeakyReLU()(layer)
        layer = Dropout(0.25)(layer)
        layer = Dense(50)(layer)
        layer = Dropout(0.5)(layer)

        output = Dense(self.num_classes, activation='softmax')(layer)
        model = Model([inputs_2[0], inputs_2[1], inputs_1[0], inputs_1[1]], output)
        self.model = model
