from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dropout, Dense, LSTM, Reshape, Bidirectional, add, \
    GlobalAveragePooling2D, BatchNormalization, LeakyReLU
from keras_self_attention import SeqSelfAttention

from source.base_dl_model import BaseDLModel
from source.models.dense_layers.dense_layers import create_dense_network


class BiLSTM(BaseDLModel):
    num_of_neurons = [[50, 50],
                      [50, 50],
                      [150, 50]]
    bidirectional_cells = 128

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shape]

        layers = [Bidirectional(LSTM(128, recurrent_dropout=0.1, dropout=0.1)(layer)) for layer in inputs]
        layers = [Dense(128)(layer) for layer in layers]
        layer = concatenate([layer for layer in layers])
        layer = Dropout(0.2)(layer)
        output = Dense(self.num_classes, activation='softmax')(layer)

        model = Model(inputs, output)
        self.model = model

    def __str__(self):
        return 'BiLSTM'
