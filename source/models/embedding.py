from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant

# define model
from source.base_dl_model import BaseDLModel


class Embedd(BaseDLModel):

    def __init__(self, **parameters):
        super(Embedd, self).__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [Flatten()(input) for input in inputs]
        layer = concatenate([layer for layer in layers])
        embedding_layer = Embedding(2629, 50)(layer)
        layer = LSTM(128, return_sequences=False)(embedding_layer)
        output = Dense(self.num_classes, activation='softmax')(layer)
        model = Model(inputs, output)
        self.model = model
