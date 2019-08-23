from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dense, LSTM, Dropout

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network


class CNN1DRNN(BaseDLModel):
    num_of_neurons = [[100],
                      [100],
                      [50]]
    convolution_filters = [32, 64]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [Flatten()(layer) for layer in inputs]
        layers = [create_cnn_network(layer, [32, 32, 64], kernels=[9, 15, 18]) for layer in layers]
        # layer_1 = concatenate([layer for layer in layers_1])
        # layer_1 = Reshape((int(layer_1.shape[1]), 1))(layer_1)
        layers = [LSTM(128, return_sequences=False, recurrent_dropout=0.2, dropout=0.2)(layer) for layer in layers]
        layers = [Dense(200)(layer) for layer in layers]
        layer = concatenate([layer for layer in layers])
        layer = Dropout(0.5)(layer)
        output = Dense(self.num_classes, activation='softmax')(layer)
        model = Model(inputs, output)
        self.model = model
