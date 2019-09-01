import keras
from keras.optimizers import Adam, RMSprop
import numpy as np

from source.grid_search import data
from source.models.cnn_and_lstm import CNNLSTM
from split_data import cut_classes, label_mappings, save_data, create_csv
from utils.settings import NUM_OF_CLASSES, NS_STANDARDIZED_VALID_DIR, NS_STANDARDIZED_TRAIN_DIR, \
    NS_STANDARDIZED_TEST_DIR, NS_NORMALIZED_TRAIN_DIR, NS_NORMALIZED_VALID_DIR, NS_NORMALIZED_TEST_DIR
from source.data_reader import load_all_data, create_3d_array, create_4d_array
from source.models import CNN, RNNLSTM, ANN, CNNRNN, RNNGRU
from source.plotting_predictions import plot_confidence, plot_classes
from utils.utilites import calculate_weights, smooth_labels


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

parameters = {'epochs': 150,
              'batch_size': 64,
              'optimizer': Adam,
              'learning_rate': 0.01,
              'num_classes': NUM_OF_CLASSES,
              'save_dir': f'./model_weights/ns/standardized_data/cnn_deep/top/Adam/{NUM_OF_CLASSES}'}
              # 'load_dir': f'./model_weights/ns/standardized_data/lstm/rmsprop/{NUM_OF_CLASSES}/25-1.987-0.524-1.442-0.584.hdf5'}


if __name__ == '__main__':

    X_train, y_train_cate, X_valid, y_valid_cate, X_test, y_test_cate, weight_class = data(standardized=standardized,
                                                                                           num_of_classes=NUM_OF_CLASSES,
                                                                                           top_classes=top_classes)

    smooth_labels(y_train_cate, smooth_factor)

    dnn = CNN(**parameters)
    # dnn.load_model(parameters["load_dir"])
    dnn.train(X_train,
              y_train_cate,
              X_valid,
              y_valid_cate,
              weight_class)

    y_pred = dnn.predict(X_valid)

    # X_test, y_test = load_all_data(TEST_DIR)
    # y_test = [dict_mapping[label] for label in y_test]
    #
    # y_test_cate = keras.utils.to_categorical(y_test, NUM_OF_CLASSES)
    #
    # X_test.pop(1)
    # X_test = X_test[:3]
    #
    # eval = dnn.model.evaluate(X_test, y_test_cate, batch_size=64)
    # print(f'Accuracy is {eval[1]}')
    # plot_confidence(y_valid, y_pred)
    # plot_classes(y_valid, y_pred)

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

