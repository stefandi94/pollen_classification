from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dense, LSTM, Reshape, Masking, Dropout, BatchNormalization, LeakyReLU, \
    Activation, add, Add, GRU

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers import create_cnn_network


class RNNGRU(BaseDLModel):

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs_1 = [Input(input_shape) for input_shape in self.rnn_shape]

        layers = [GRU(256, recurrent_dropout=0.1, dropout=0.1)(layer) for layer in inputs_1]
        layers = [Dense(128)(layer) for layer in layers]
        layers = concatenate([layer for layer in layers])

        layers = Dropout(0.2)(layers)
        outputs = Dense(self.num_classes, activation='softmax')(layers)
        model = Model(inputs_1, outputs)
        self.model = model

    def __str__(self):
        return 'GRU'
