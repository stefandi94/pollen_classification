from source.models import ANN, CNN, RNNGRU, BiLSTM, RNNLSTM, CNN_1_LSTM_2
from source.models.cnn_and_lstm import CNNLSTM


def get_model(model_name):
    if model_name == 'ANN':
        model = ANN
    elif model_name == 'CNN':
        model = CNN
    elif model_name == 'GRU':
        model = RNNGRU
    elif model_name == 'BiLSTM':
        model = BiLSTM
    elif model_name == 'LSTM':
        model = RNNLSTM
    elif model_name == 'CNNRNN':
        model = CNN_1_LSTM_2
    elif model_name == 'CNNLSTM':
        model = CNNLSTM

    return model
