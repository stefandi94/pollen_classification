from typing import Any

from keras.layers import concatenate, Dropout, Dense, Input, Activation, Reshape
from keras.models import Model

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network


class CNN(BaseDLModel):
    convolution_filters = [32, 64, 64, 128]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.cnn_shape]

        # inputs_1 = [Reshape((20, 120, 1))(inputs[0]),
        #             Reshape((4, 24, 1))(inputs[1]),
        #             Reshape((4, 32, 1))(inputs[2])]

        layers = [create_cnn_network(layer, self.convolution_filters) for layer in inputs]
        layers = [Activation('relu')((Dense(100)(layer))) for layer in layers]
        layer = concatenate([layer for layer in layers])

        layer = Dropout(0.2)(layer)
        output = Dense(self.num_classes, activation='softmax')(layer)

        model = Model(inputs, output)
        self.model = model

    def __str__(self):
        return 'CNN'
