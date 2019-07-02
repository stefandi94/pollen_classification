from typing import Any

from keras import Input, Model
from keras.layers import concatenate, Dropout, Dense

from source.base_dl_model import BaseDLModel
from .layers import create_wide_residual_network


class WideResNet(BaseDLModel):
    dropout = 0.2

    def __init__(self,
                 k: int,
                 N: int,
                 **parameters: Any) -> None:
        """
        :param k: depth of network
        :param N: depth of layers
        :param parameters:
        """

        super().__init__(**parameters)

        self.k = k
        self.N = N

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [create_wide_residual_network(self.k, self.N, layer, self.dropout) for layer in inputs]
        layers = concatenate([layer for layer in layers])
        layers = Dropout(0.5)(layers)

        output = Dense(self.num_classes, activation='softmax')(layers)

        model = Model(inputs, output)
        self.model = model
