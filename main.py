import keras
from keras.optimizers import Adam, RMSprop
import numpy as np

from source.grid_search import data
from source.models.cnn_and_lstm import CNNLSTM
from split_data import cut_classes, label_mappings, save_data, create_csv
from utils.settings import NUM_OF_CLASSES, NS_STANDARDIZED_VALID_DIR, NS_STANDARDIZED_TRAIN_DIR, \
    NS_STANDARDIZED_TEST_DIR, NS_NORMALIZED_TRAIN_DIR, NS_NORMALIZED_VALID_DIR, NS_NORMALIZED_TEST_DIR
from source.data_reader import load_all_data, create_3d_array, create_4d_array
from source.models import CNN, RNNLSTM, ANN, CNNRNN, RNNGRU, BiLSTM
from source.plotting_predictions import plot_confidence, plot_classes
from utils.utilites import calculate_weights, smooth_labels, count_values

smooth_factor = 0.1
shapes = dict(input_shape_1=(20, 120),
              input_shape_2=(4, 24),
              input_shape_3=(4, 32))

standardized = True
normalized = False
NUM_OF_CLASSES = 50
top_classes = True
if standardized:
    TRAIN_DIR = NS_STANDARDIZED_TRAIN_DIR
    VALID_DIR = NS_STANDARDIZED_VALID_DIR
    TEST_DIR = NS_STANDARDIZED_TEST_DIR
elif normalized:
    TRAIN_DIR = NS_NORMALIZED_TRAIN_DIR
    VALID_DIR = NS_NORMALIZED_VALID_DIR
    TEST_DIR = NS_NORMALIZED_TEST_DIR

parameters = {'epochs': 1,
              'batch_size': 64,
              'optimizer': 'adam',
              'num_classes': NUM_OF_CLASSES,
              'save_dir': f'./model_weights/ns/standardized_data/cnn_deep/top/Adam/{NUM_OF_CLASSES}',
              'load_dir': f'./model_weights/ns/standardized_data/cnn_deep/top/Adam/50/5-2.672-0.350-2.307-0.377.hdf5'}


if __name__ == '__main__':

    X_train, y_train, X_valid, y_valid, X_test, y_test, weight_class = data(standardized=standardized,
                                                                            num_of_classes=NUM_OF_CLASSES,
                                                                            top_classes=top_classes)

    y_train_cate = keras.utils.to_categorical(y_train, NUM_OF_CLASSES)
    y_valid_cate = keras.utils.to_categorical(y_valid, NUM_OF_CLASSES)
    y_test_cate = keras.utils.to_categorical(y_test, NUM_OF_CLASSES)

    smooth_labels(y_train_cate, smooth_factor)

    dnn = BiLSTM(**parameters)
    dnn.train(X_train,
              y_train_cate,
              X_valid,
              y_valid_cate,
              weight_class=weight_class,
              lr_type='cosine')
    # dnn.load_model(parameters["load_dir"])

    y_pred = dnn.predict(X_valid)

    eval = dnn.model.evaluate(X_test, y_test_cate, batch_size=64)
    print(f'Accuracy is {eval[1]}')
    plot_confidence(y_test, y_pred, parameters['save_dir'], num_classes=NUM_OF_CLASSES)
    plot_classes(y_test, y_pred, parameters['save_dir'], NUM_OF_CLASSES)



