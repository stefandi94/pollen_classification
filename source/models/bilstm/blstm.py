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
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [Flatten()(layer) for layer in inputs]
        layers = [create_dense_network(layer, num_of_neurons=self.num_of_neurons[2]) for layer in layers]
        layers = [Reshape((int(layer.shape[1]), 1))(layer) for layer in layers]

        lstms = [Bidirectional(LSTM(128, return_sequences=False, recurrent_dropout=0.25, dropout=0.25))(layer) for layer in layers]

        layers = [create_dense_network(layer, num_of_neurons=self.num_of_neurons[2]) for layer in lstms]
        lay = concatenate([layer for layer in layers])

        # attention = SeqSelfAttention(attention_activation='sigmoid')(lstm1)
        # attention = Flatten()(attention)
        # attention = Dropout(0.2)(attention)
        # attention = BatchNormalization()(attention)
        # attention = LeakyReLU()(attention)

        attention = Dense(50)(lay)
        attention = Dropout(0.5)(attention)
        output = Dense(self.num_classes, activation="softmax")(attention)

        model = Model(inputs, output)
        self.model = model
