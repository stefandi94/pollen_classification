from typing import Any

from keras.layers import concatenate, Dropout, Dense, Input, Lambda, add, Activation
from keras.models import Model

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network
from source.models.dense_layers.dense_layers import create_dense_network


class CNN(BaseDLModel):
    # num_of_neurons = [[200, 50],
    #                   [200, 50]]
    convolution_filters = [32, 64, 128, 256]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]

        layers = [create_cnn_network(layer, self.convolution_filters) for layer in inputs]
        layer = concatenate([layer for layer in layers])

        layer = Dropout(0.5)(layer)
        output = Dense(self.num_classes, activation='softmax')(layer)

        model = Model(inputs, output)
        self.model = model
