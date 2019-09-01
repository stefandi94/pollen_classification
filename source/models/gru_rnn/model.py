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

        layers_1 = [GRU(512, recurrent_dropout=0.0, dropout=0.0)(layer) for layer in inputs_1]
        layers_1 = concatenate([layer for layer in layers_1])

        layers = Dropout(0.2)(layers_1)
        outputs = Dense(self.num_classes, activation='softmax')(layers)
        model = Model(inputs_1, outputs)
        self.model = model

    def __str__(self):
        return 'GRU'
