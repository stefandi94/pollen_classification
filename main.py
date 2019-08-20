from time import time
from datetime import datetime
import keras
from keras.optimizers import Adam, Adagrad
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, accuracy_score
import numpy as np
from settings import NUM_OF_CLASSES, NORMALIZED_TRAIN_DIR, NORMALIZED_VALID_DIR, STANDARDIZED_TRAIN_DIR, \
    STANDARDIZED_VALID_DIR
from source.data_reader import load_all_data, create_3d_array, create_4d_array
from source.models import ResidualNet, ANNCNN, CNN
from source.plotting_predictions import plot_confidence, plot_classes
from split_data import flatten_data
from utils.utilites import calculate_weights, smooth_labels
from xgboost import XGBClassifier

parameters = dict(epochs=20, batch_size=64, optimizer=Adam, learning_rate=0.007,
                  save_dir=f'./model_weights/cnn/standardized_data/adam/')
                  # load_dir='./model_weights/cnn/4-2.794-0.308-2.256-0.365.hdf5')

smooth_factor = 0.1
rnn_shapes = dict(input_shape_1=(20, 120, 1),
                  input_shape_2=(4, 24, 1),
                  input_shape_3=(4, 32, 1),
                  input_shape_4=(4, 1, 1))

if __name__ == '__main__':

    X_train, y_train = load_all_data(STANDARDIZED_TRAIN_DIR)
    X_valid, y_valid = load_all_data(STANDARDIZED_VALID_DIR)

    X_train.pop(1)  # remove 1x1 feature
    X_valid.pop(1)

    for index in range(len(X_train)):
        if len(X_train[index].shape) < 3:
            X_train[index] = create_3d_array(X_train[index])

        if len(X_valid[index].shape) < 3:
            X_valid[index] = create_3d_array(X_valid[index])

    for index in range(len(X_train)):
        X_train[index] = create_4d_array(X_train[index])
        X_valid[index] = create_4d_array(X_valid[index])

    weight_class = calculate_weights(y_train)

    y_train_cate = keras.utils.to_categorical(y_train, NUM_OF_CLASSES)
    y_valid_cate = keras.utils.to_categorical(y_valid, NUM_OF_CLASSES)

    smooth_labels(y_train_cate, smooth_factor)

    dnn = CNN(**parameters)
    dnn.rnn_shapes = rnn_shapes
    # dnn.load_model(parameters["load_dir"])
    dnn.train(X_train, y_train_cate,
              X_valid, y_valid_cate,
              weight_class)

    y_pred = dnn.predict(X_valid)
    plot_confidence(y_valid, y_pred)
    plot_classes(y_valid, y_pred)

    # new_x_train = flatten_data(X_train)
    # new_x_valid = flatten_data(X_valid)
    #
    # print(f'Started PCA fitting at {datetime.now().time()}')
    # pca = PCA(n_components=150)
    # new_x_train = pca.fit_transform(new_x_train, y_train)
    # new_x_valid = pca.transform(new_x_valid)
    #
    # xgb = XGBClassifier(max_depth=5,
    #                     learning_rate=0.06,
    #                     n_estimators=500,
    #                     verbosity=2,
    #                     silent=False,
    #                     n_jobs=-1,
    #                     nthread=1)
    #
    # print(f'Fitting started at {datetime.now().time()}')
    # xgb.fit(new_x_train, y_train)
    # print(f'Fitting finished at {datetime.now().time()}')
    # y_pred = xgb.predict(new_x_valid)
    #
    # print(f'Accuracy score is {accuracy_score(y_valid, y_pred)}')
    # print(f'Recall score is: {recall_score(y_valid, y_pred, average="macro")}')
    # print(f'F1 score is: {f1_score(y_valid, y_pred, average="macro")}')

