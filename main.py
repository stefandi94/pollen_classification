import os.path as osp

import keras

# from dense import Dense, Dense_net
from settings import MODEL_DIR
from source.data_reader import load_all_data
from source.models import BiLSTM
from utils.utilites import calculate_weights, smooth_labels

smooth_factor = 0.03
learning_rate = 0.01
weigh_decay = 0.0005
model_path = osp.join(MODEL_DIR, 'encoder', f'{learning_rate}')

# load_path = "/home/stefan/PycharmProjects/aaaa/models/encoder/0.01/  1-3.055-0.174-\n2.663-0.255.hdf5"

epochs = 100
batch_size = 256


parameters = {'batch_size': batch_size,
              'model_path': model_path,
              'epochs': epochs}

warmup_epoch = 15
hold_base_rate = 30
warmup_learning_rate = 0.001


if __name__ == '__main__':

    X_train, y_train = load_all_data('train')
    X_valid, y_valid = load_all_data('valid')

    X_train = X_train[0]
    X_valid = X_valid[0]

    weight_class = calculate_weights(y_train)
    y_train = keras.utils.to_categorical(y_train, 50)
    y_valid = keras.utils.to_categorical(y_valid, 50)

    smooth_labels(y_train, smooth_factor)

    cnn = BiLSTM(**parameters)
    cnn.train(X_train, y_train, X_valid, y_valid, weight_class, learning_rate)