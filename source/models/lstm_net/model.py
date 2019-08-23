from typing import Any

from keras import Input, Model
from keras.layers import Flatten, concatenate, Dense, LSTM, Reshape, Masking, Dropout, BatchNormalization, LeakyReLU, \
    Activation
from keras_self_attention import SeqSelfAttention

from source.base_dl_model import BaseDLModel
from source.models.dense_layers.dense_layers import create_dense_network


class RNNLSTM(BaseDLModel):

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        # layers = [Flatten()(layer) for layer in inputs]
        # layers = [create_dense_network(layer, num_of_neurons=[50]) for layer in layers]
        # layers = [Reshape((int(layer.shape[1]), 1))(layer) for layer in layers]

        layers = [LSTM(256, return_sequences=True, recurrent_dropout=0.25, dropout=0.25)(layer) for layer in inputs]
        # layers = concatenate([layer for layer in lstms])
        layers = [LSTM(128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25)(layer) for layer in layers]
        layers = [SeqSelfAttention(attention_activation='sigmoid')(layer) for layer in layers]
        layers = [Dense(256)(layer) for layer in layers]
        layers = [Activation('relu')(layer) for layer in layers]
        layers = [Dropout(0.5)(layer) for layer in layers]
        layers = [BatchNormalization()(layer) for layer in layers]
        layers = [Flatten()(layer) for layer in layers]
        layers = concatenate([layer for layer in layers])
        # layer = LeakyReLU()(layer)

        # layers = Dense(50)(layer)
        layers = Dropout(0.5)(layers)
        output = Dense(self.num_classes, activation="softmax")(layers)

        model = Model(inputs, output)
        self.model = model
