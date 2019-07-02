from typing import Any

from keras.layers import Input, Dropout, concatenate, Dense
from keras.models import Model

from source.base_dl_model import BaseDLModel
from source.models.mobile_net.layers import mobile_net_v2


class MobileNetV2(BaseDLModel):

    def __init__(self,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

    def build_model(self) -> None:
        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [mobile_net_v2(layer) for layer in inputs]

        x = concatenate([layer for layer in layers])
        x = Dropout(0.5)(x)

        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, output)
        self.model = model
