from typing import Any

from keras import Input, Model
from keras.layers import concatenate, Dense, LSTM, Reshape, Dropout

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers import create_cnn_network


class CNNLSTM(BaseDLModel):
    convolution_filters = [32, 64, 128]

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs_1 = [Input(input_shape) for input_shape in self.rnn_shape]
        inputs_2 = [Reshape((20, 120, 1))(inputs_1[0]),
                    Reshape((4, 24, 1))(inputs_1[1]),
                    Reshape((4, 32, 1))(inputs_1[2])]

        layers_1 = [LSTM(128, recurrent_dropout=0.1, dropout=0.1)(layer) for layer in inputs_1]
        layers_1 = concatenate([layer for layer in layers_1])

        layers_2 = [create_cnn_network(layer, self.convolution_filters, dropout=0.0) for layer in inputs_2]
        layers_2 = concatenate([layer for layer in layers_2])

        layers = concatenate([layers_1, layers_2])
        layers = Dropout(0.5)(layers)
        outputs = Dense(self.num_classes, activation='softmax')(layers)
        model = Model(inputs_1, outputs)
        self.model = model


