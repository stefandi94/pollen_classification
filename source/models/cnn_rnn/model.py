from typing import Any

from keras import Input, Model
from keras.layers import concatenate, Dense, LSTM, Reshape, Dropout

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers.cnn_layers import create_cnn_network


class CNN_1_LSTM_2(BaseDLModel):
    rnn_shapes = [(20, 120)]
    cnn_shapes = [(4, 32), (4, 24)]
    convolution_filters = [64, 128, 256]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        # inputs_1 = [Input((4, 32)), Input((4, 24))]
        inputs_1 = [Input(input_shape) for input_shape in self.rnn_shape]
        inputs_2 = [Reshape((20, 120, 1))(inputs_1[0])]

        # inputs_2 = [Input((20, 120))]

        layers_1 = [LSTM(128, return_sequences=False, recurrent_dropout=0.2)(layer) for layer in inputs_1]
        layers_2 = create_cnn_network(inputs_2[0], self.convolution_filters)

        layers_1 = [Dense(128)(layer) for layer in layers_1]
        layers_2 = Dense(128)(layers_2)

        layer_1 = concatenate([layer for layer in layers_1])
        layer = concatenate([layer_1, layers_2])

        flatten = Dropout(0.5)(layer)
        output = Dense(self.num_classes, activation='softmax')(flatten)

        model = Model(inputs_1, output)
        self.model = model

    def __str__(self):
        return 'CNNRNN'
