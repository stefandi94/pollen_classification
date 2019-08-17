from keras import Input, Model
from keras.layers import Dropout, concatenate, Dense
from keras.optimizers import Adam

from source.base_dl_model import BaseDLModel
from source.models.inception_net.layers import create_inception_model


class Inception(BaseDLModel):

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [create_inception_model(layer) for layer in inputs]

        output = concatenate([layer[0] for layer in layers])
        output = Dropout(0.5)(output)
        output = Dense(self.num_classes, activation='softmax')(output)

        aux_output_2 = concatenate([layer[1] for layer in layers])
        aux_output_2 = Dropout(0.5)(aux_output_2)
        aux_output_2 = Dense(self.num_classes, activation='softmax')(aux_output_2)

        aux_output_1 = concatenate([layer[2] for layer in layers])
        aux_output_1 = Dropout(0.5)(aux_output_1)
        aux_output_1 = Dense(self.num_classes, activation='softmax')(aux_output_1)

        model = Model(inputs, [output, aux_output_2, aux_output_1])
        model.compile(loss=['categorical_crossentropy',
                            'categorical_crossentropy',
                            'categorical_crossentropy'],
                      loss_weights=[1, 0.6, 0.4],
                      optimizer=Adam(), metrics=['accuracy'])
        self.model = model
