import keras
from keras.optimizers import Adam

from settings import NUM_OF_CLASSES
from source.data_reader import load_all_data
from source.models import BiLSTM
from source.plotting_predictions import plot_confidence
from utils.utilites import calculate_weights, smooth_labels

parameters = dict(epochs=600, batch_size=256, optimizer=Adam, learning_rate=0.005,
                  save_dir='./model_weights/bilst_separated/0.005',
                  load_dir='./model_weights/bilst_separated/0.005/25-1.931-0.465-1.677-0.510.hdf5')

smooth_factor = 0.08
rnn_shapes = dict(input_shape_1=(4, 32, 1),
                  input_shape_2=(4, 24, 1),
                  input_shape_3=(20, 120, 1))
                  # input_shape_4=(5, 1, 1))

if __name__ == '__main__':

    X_train, y_train = load_all_data('train')
    X_valid, y_valid = load_all_data('valid')

    weight_class = calculate_weights(y_train)

    y_train_cate = keras.utils.to_categorical(y_train, NUM_OF_CLASSES)
    y_valid_cate = keras.utils.to_categorical(y_valid, NUM_OF_CLASSES)

    smooth_labels(y_train_cate, smooth_factor)

    dnn = BiLSTM(**parameters)
    dnn.rnn_shapes = rnn_shapes
    # dnn.load_model(parameters["load_dir"])
    dnn.train(X_train, y_train_cate, X_valid, y_valid_cate, weight_class)
    # y_pred = dnn.predict(X_valid)
    # plot_confidence(y_valid, y_pred)
