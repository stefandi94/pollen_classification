from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dense, LSTM, Reshape, Masking, Dropout, BatchNormalization, LeakyReLU, \
    Activation, add, Add

from source.base_dl_model import BaseDLModel
from source.models.convolutional_layers import create_cnn_network


class RNNLSTM(BaseDLModel):
    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shape]

        layers = [LSTM(128, recurrent_dropout=0.1, dropout=0.1)(layer) for layer in inputs]
        layers = [Dense(128)(layer) for layer in layers]
        layer = concatenate([layer for layer in layers])
        layer = Dropout(0.2)(layer)
        output = Dense(self.num_classes, activation='softmax')(layer)

        model = Model(inputs, output)
        self.model = model


