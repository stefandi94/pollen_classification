from typing import Any, List

from keras import Input
from keras.layers import concatenate, Dropout, Dense
from keras.models import Model

from source.base_dl_model import BaseDLModel
from source.models.res_net.layers import create_residual


class ResidualNet(BaseDLModel):

    def __init__(self,
                 num_layer: int,
                 **parameters: Any) -> None:
        super().__init__(**parameters)

        self.num_layer = num_layer
        self.num_to_model = {18: self.resnet18,
                             34: self.resnet34,
                             50: self.resnet50,
                             101: self.resnet101,
                             152: self.resnet152}

        if num_layer not in self.num_to_model.keys():
            raise Exception('Model number should be 18, 34, 59, 101 or 152, try again!')

    def build_model(self) -> None:
        self.num_to_model[self.num_layer]()

    def _resnet(self,
                convx: List[int] = [32, 64, 128, 256],
                n_convx: List[int] = [1, 1, 1, 1]) -> None:
        """
        :param convx:
        :param n_convx:
        :return:
        """

        inputs = [Input(input_shape) for input_shape in self.rnn_shapes.values()]
        layers = [create_residual(layer, convx, n_convx) for layer in inputs]

        flatten = concatenate([layer for layer in layers])
        flatten = Dropout(0.5)(flatten)

        output = Dense(self.num_classes, activation='softmax')(flatten)
        model = Model(inputs, output)

        self.model = model

    def resnet18(self) -> None:
        return self._resnet()

    def resnet34(self) -> None:
        return self._resnet(n_convx=[3, 4, 6, 3])

    def resnet50(self) -> None:
        return self._resnet([256, 512, 1024, 2048],
                            [3, 4, 6, 3])

    def resnet101(self) -> None:
        return self._resnet([256, 512, 1024, 2048],
                            [3, 4, 23, 3])

    def resnet152(self) -> None:
        return self._resnet([256, 512, 1024, 2048],
                            [3, 8, 36, 3])
