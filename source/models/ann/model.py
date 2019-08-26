from typing import Any

import keras
from keras.layers import Flatten, concatenate, Dropout, Dense, Input
from keras.models import Model

from source.base_dl_model import BaseDLModel
from source.models.dense_layers.dense_layers import create_dense_network

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)


class ANN(BaseDLModel):
    num_of_neurons = [200]

    def __init__(self,
                 **parameters: Any) -> None:
        super(ANN, self).__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.shapes.values()]
        layers = [Dropout(0.25)(layer) for layer in inputs]
        layers = [create_dense_network(layer, num_of_neurons=self.num_of_neurons) for layer in layers]
        layers = [Flatten()(layer) for layer in layers]
        layer = concatenate([layer for layer in layers])

        flatten = Dropout(0.5)(layer)
        output = Dense(self.num_classes, activation='softmax')(flatten)

        model = Model(inputs, output)
        self.model = model
