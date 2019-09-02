from __future__ import print_function

import keras
import numpy as np

from source.data_reader import load_all_data, create_3d_array, create_4d_array
from split_data import cut_classes, label_mappings, save_data
from utils.settings import NS_STANDARDIZED_TRAIN_DIR, NS_NORMALIZED_VALID_DIR, NS_NORMALIZED_TEST_DIR, \
    NS_NORMALIZED_TRAIN_DIR, NS_STANDARDIZED_TEST_DIR, NS_STANDARDIZED_VALID_DIR
from utils.utilites import calculate_weights, smooth_labels


def data(standardized, num_of_classes, top_classes=True):
    if standardized:
        TRAIN_DIR = NS_STANDARDIZED_TRAIN_DIR
        VALID_DIR = NS_STANDARDIZED_VALID_DIR
        TEST_DIR = NS_STANDARDIZED_TEST_DIR
    else:
        TRAIN_DIR = NS_NORMALIZED_TRAIN_DIR
        VALID_DIR = NS_NORMALIZED_VALID_DIR
        TEST_DIR = NS_NORMALIZED_TEST_DIR

    X_train, y_train = load_all_data(TRAIN_DIR)
    X_valid, y_valid = load_all_data(VALID_DIR)
    X_test, y_test = load_all_data(TEST_DIR)

    X_train.pop(1)  # remove 1x1 feature
    X_valid.pop(1)
    X_test.pop(1)

    X_train = X_train[:3]
    X_valid = X_valid[:3]
    X_test = X_test[:3]

    for index in range(len(X_train)):
        if len(X_train[index]) < 3:
            X_train[index] = create_3d_array(X_train[index])

        if len(X_valid[index]) < 3:
            X_valid[index] = create_3d_array(X_valid[index])

        if len(X_test[index]) < 3:
            X_test[index] = create_3d_array(X_test[index])

    # for index in range(len(X_train)):
    #     X_train[index] = create_4d_array(X_train[index])
    #     X_valid[index] = create_4d_array(X_valid[index])
    #     X_test[index] = create_4d_array(X_test[index])

    X_train, y_train, classes_to_take = cut_classes(data=X_train,
                                                    labels=y_train,
                                                    num_of_class=num_of_classes,
                                                    top=top_classes)
    X_valid, y_valid, _ = cut_classes(data=X_valid,
                                      labels=y_valid,
                                      top=True,
                                      classes_to_take=classes_to_take)

    X_test, y_test, _ = cut_classes(data=X_test,
                                    labels=y_test,
                                    top=True,
                                    classes_to_take=classes_to_take)

    dict_mapping = label_mappings(classes_to_take)
    save_data(dict_mapping, data_path=TRAIN_DIR, filename='mapping')
    y_train = [dict_mapping[label] for label in y_train]
    y_valid = [dict_mapping[label] for label in y_valid]
    y_test = [dict_mapping[label] for label in y_test]

    weight_class = calculate_weights(y_train)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, weight_class

