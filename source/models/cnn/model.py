from typing import Any

from keras.layers import concatenate, Dropout, Dense, Input, Lambda, add, Activation
from keras.models import Model

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network
from source.models.dense_layers.dense_layers import create_dense_network


class CNN(BaseDLModel):
    # num_of_neurons = [[200, 50],
    #                   [200, 50]]
    convolution_filters = [64, 128, 256]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]

        # Create dense_layers layers first
        # layers = [create_dense_network(layer, num_of_neurons=self.num_of_neurons[index]) for index, layer in
        #           enumerate(inputs)]

        # Then create convolutional layers
        layers = [create_cnn_network(layer, self.convolution_filters) for layer in inputs]

        layers = [Dropout(0.5)(layer) for layer in layers]
        layers = [Dense(self.num_classes, activation='softmax')(layer) for layer in layers]
        # layer = concatenate([layer for layer in layers])
        output = add([layer for layer in layers])
        output = Lambda(lambda x: x * 3)(output)

        model = Model(inputs, output)
        self.model = model
