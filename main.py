import os
import pickle

from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np

from source.data_loader import data
from settings import NS_STANDARDIZED_VALID_DIR, NS_STANDARDIZED_TRAIN_DIR, \
    NS_STANDARDIZED_TEST_DIR, NS_NORMALIZED_TRAIN_DIR, NS_NORMALIZED_VALID_DIR, NS_NORMALIZED_TEST_DIR
from source.models import BiLSTM, ANN
from source.plotting_predictions import plot_confidence, plot_classes, create_dict_conf, plot_confidence_per_class, \
    plot_confusion_matrix, plot_history
from utils.split_data import save_data
from utils.utilites import smooth_labels

smooth_factor = 0.0
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

# load_dir = '/mnt/hdd/PycharmProjects/pollen_classification/new_weights/ns/normalized/smooth_factor_0.1/optimizer_adam' \
#            '/learning_rate_type_cosine/model_name_CNNRNN/ '
load_dir = '/mnt/hdd/PycharmProjects/pollen_classification/new_weights/ns/standard_normal/smooth_factor_0.0/optimizer_rmsprop/learning_rate_type_cosine/model_name_GRU/'
parameters = {'epochs': 30,
              'batch_size': 256,
              'optimizer': 'adam',
              'num_classes': NUM_OF_CLASSES,
              'save_dir': f'{os.path.join(load_dir, str(NUM_OF_CLASSES))}',
              'load_dir': f'{os.path.join(load_dir, "10-1.744-0.515-1.631-0.549.hdf5")}'}


if __name__ == '__main__':

    X_train, y_train, X_valid, y_valid, X_test, y_test, weight_class, dict_mapping = data(standardized=True,
                                                                                          num_of_classes=NUM_OF_CLASSES,
                                                                                          top_classes=top_classes,
                                                                                          ns=True,
                                                                                          create_4d_arr=False)

    y_train_cate = to_categorical(y_train, NUM_OF_CLASSES)
    y_valid_cate = to_categorical(y_valid, NUM_OF_CLASSES)
    y_test_cate = to_categorical(y_test, NUM_OF_CLASSES)

    smooth_labels(y_train_cate, smooth_factor)

    dnn = ANN(**parameters)
    dnn.load_model(parameters["load_dir"])
    # dnn.train(X_train,
    #           y_train_cate,
    #           X_valid,
    #           y_valid_cate,
    #           weight_class=weight_class,
    #           lr_type='cyclic')
    print()
    import os.path as osp
    os.makedirs('./test', exist_ok=True)
    # with open(osp.join('./test', 'mapping.pckl'), 'wb') as handle:
    #     pickle.dump(dict_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    y_pred = dnn.predict(X_test)
    eval = dnn.model.evaluate(X_test, y_test_cate, batch_size=64)
    # TODO: napraviti dict mapping za 15 i 35 klasa, posto nisu sacuvani?

    # real_y = [dict_mapping[y] for y in y_test]
    # pred_y = [(dict_mapping[y[0]], y[1]) for y in y_pred]

    print(f'Accuracy is {eval[1]}')
    y_class_pred = [int(pred[0]) for pred in y_pred]
    conf_matrix = confusion_matrix(y_test, y_class_pred)
    true_conf, true_dicti, false_conf, false_dicti = create_dict_conf(y_test, y_pred, NUM_OF_CLASSES)

    # plot_confusion_matrix(y_test, y_class_pred, list(dict_mapping.keys()), './test')
    # plot_confusion_matrix(y_test, y_class_pred, list(dict_mapping.keys()), parameters['save_dir'], normalize=True)
    plot_confidence(true_conf, false_conf, parameters['save_dir'], show_plot=False)
    plot_classes(y_test, y_pred, parameters['save_dir'], NUM_OF_CLASSES, show_plot=False)
    plot_confidence_per_class(true_dicti, false_dicti, NUM_OF_CLASSES, parameters['save_dir'], show_plot=False)
    plot_history(log_path=parameters['save_dir'])
