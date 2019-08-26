from typing import List

from keras import Input

from source.base_dl_model import BaseDLModel
from source.models import CNN, RNNLSTM


class MultiModal(BaseDLModel):

    def __init__(self, models: List, weights: List):
        self.modes = models
        self.weights = weights

        super(MultiModal, self).__init__()

    def build_model(self, **parameters) -> None:
        input_1 = Input(shape=(20, 120))
        input_2 = Input(shape=(4, 24))
        input_3 = Input(shape=(4, 32))

        model_1 = CNN(**parameters, shape=(20, 120))
        model_2 = CNN(**parameters, shape=(4, 24))
        model_3 = CNN(**parameters, shape=(4, 32))

        model_4 = RNNLSTM(**parameters, shape=(20, 120))
        model_5 = RNNLSTM(**parameters, shape=(4, 24))
        model_6 = RNNLSTM(**parameters, shape=(4, 32))
