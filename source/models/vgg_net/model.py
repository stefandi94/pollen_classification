from typing import Any

from keras import Input, Model
from keras.layers import Dense, GaussianNoise, Dropout, concatenate

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network
from source.models.dense_layers.dense_layers import dense_layer


class VGG(BaseDLModel):

    def __init__(self,
                 vgg_type: int,
                 **parameters: Any) -> None:

        super().__init__(**parameters)

        self.vgg_type = vgg_type
        if self.vgg_type not in [16, 19]:
            raise Exception("Please choose between vgg_16 and vgg_19 networks")

    def build_model(self) -> None:

        if self.vgg_type == 16:
            n_filters = [64, 128, 256]
        else:
            n_filters = [64, 64, 128, 128, 256, 256]

        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [GaussianNoise(0.02)(layer) for layer in inputs]
        layers = [create_cnn_network(layer, n_filters) for layer in layers]
        layers = [Dropout(0.25)(layer) for layer in layers]
        layer = concatenate([layer for layer in layers])

        dense1 = dense_layer(layer, 512, dropout=0.5)
        dense2 = dense_layer(dense1, 256, dropout=0.5)

        output = Dense(self.num_classes, activation='softmax')(dense2)
        self.model = Model(inputs, output)
