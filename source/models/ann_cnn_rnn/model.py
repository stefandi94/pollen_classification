from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dense, LSTM, Reshape

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network
from source.models.dense_layers.dense_layers import create_dense_network


class ANNCNNRNN(BaseDLModel):
    num_of_neurons = [[200, 100, 50],
                      [200, 100, 50],
                      [200, 100, 50]]
    convolution_filters = [32, 64, 128]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]

        layers_1 = [Flatten()(layer) for layer in inputs]
        layers_1 = [create_dense_network(layer, num_of_neurons=[100, 50]) for layer in layers_1]
        layers_1 = [Reshape((int(layer.shape[1]), 1))(layer) for layer in layers_1]
        layers_1 = [LSTM(128, return_sequences=False, recurrent_dropout=0.25, dropout=0.25)(layer)
                    for layer in layers_1]

        layer_1 = concatenate([layer for layer in layers_1])

        layers_2 = [create_cnn_network(layer, self.convolution_filters) for layer in inputs]
        layer_2 = concatenate([layer for layer in layers_2])

        layers_3 = [create_dense_network(layer, num_of_neurons=[300, 200, 100, 50]) for layer in layers_1]
        layer_3 = concatenate([layer for layer in layers_3])

        layer = concatenate([layer_1, layer_2, layer_3])

        output = Dense(self.num_classes, activation='softmax')(layer)
        model = Model(inputs, output)
        self.model = model
