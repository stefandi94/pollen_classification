import keras
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from settings import NUM_OF_CLASSES
from source.data_reader import load_all_data
from source.models import RNNLSTM, BiLSTM, CNNRNN, CNN
from source.models.ann_cnn_rnn import ANNCNNRNN
from source.plotting_predictions import plot_confidence
from source.utilites import calculate_weights, smooth_labels

parameters = dict(epochs=600, batch_size=256, optimizer=Adam, learning_rate=0.005,
                  save_dir='./model_weights/bilst_separated/0.005')
                  # load_dir='./model_weights/rnn_lstm_separated/0.005/25-1.697-0.513-1.623-0.518.hdf5')

smooth_factor = 0.08
rnn_shapes = dict(input_shape_1=(4, 32, 1),
                  input_shape_2=(4, 24, 1),
                  input_shape_3=(20, 120, 1))
                  # input_shape_4=(5, 1, 1))

if __name__ == '__main__':

    X_train, y_train = load_all_data('train')
    X_valid, y_valid = load_all_data('valid')

    weight_class = calculate_weights(y_train)

    y_train = keras.utils.to_categorical(y_train, NUM_OF_CLASSES)
    y_valid = keras.utils.to_categorical(y_valid, NUM_OF_CLASSES)

    smooth_labels(y_train, smooth_factor)

    dnn = BiLSTM(**parameters)
    dnn.rnn_shapes = rnn_shapes
    # dnn.model.load_model(parameters["load_dir"])
    dnn.train(X_train, y_train, X_valid, y_valid, weight_class)
    # y_pred = dnn.predict(X_valid)
    # plot_confidence(y_valid, y_pred)
