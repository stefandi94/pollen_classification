import os
import os.path as osp
import pickle

import glob2
import numpy as np
from keras.utils import to_categorical

from settings import WEIGHTS_DIR, NS_WEIGHTS_DIR
from source.data_loader import data
from source.get_model import get_model
from source.plotting_predictions import plot_confidence, plot_classes, create_dict_conf, plot_confidence_per_class, \
    plot_confusion_matrix, plot_history
from utils.utilites import smooth_labels


def search():
    best_parameters = {}
    acc = 0
    epochs = 10
    batch_size = 64
    grid = {'data_type': ['standard_normal', 'normalized'],
            'num_of_classes': [50],  # 30
            'smooth_factor': [0.1],  #
            'optimizer': ['adam', 'rmsprop'],
            'learning_rate': ['cosine', 'cyclic'],
            'models': ['CNNRNN', 'CNNLSTM']}
    # 'models': ['ANN', 'CNN', 'CNNRNN', 'GRU', 'BiLSTM', 'CNNLSTM']}
    # Finished models: ANN, LSTM, GRU, BiLSTM., on training: CNN. Do CNNRNN and CNNLSTM on NS data

    for data_type in grid['data_type']:
        # for class_type in grid['class_types']:
        for num_classes in grid['num_of_classes']:
            X_train, y_train, X_valid, y_valid, X_test, y_test, weight_class, dict_mapping = data(
                standardized=data_type,
                num_of_classes=num_classes,
                ns=True,
                create_4d_arr=False)

            y_train_cate = to_categorical(y_train, num_classes)
            y_valid_cate = to_categorical(y_valid, num_classes)
            y_test_cate = to_categorical(y_test, num_classes)

            for smooth_factor in grid['smooth_factor']:
                smooth_labels(y_train_cate, smooth_factor)

                for optimizer in grid['optimizer']:
                    for lr_type in grid['learning_rate']:
                        for model_name in grid['models']:
                            print('Stared new fitting')
                            save_dir = f'/mnt/hdd/PycharmProjects/pollen_classification/new_weights/ns/{data_type}' \
                                       f'/smooth_factor_{smooth_factor}' \
                                       f'/optimizer_{optimizer}' \
                                       f'/learning_rate_type_{lr_type}/' \
                                       f'model_name_{model_name}/'

                            os.makedirs(save_dir, exist_ok=True)
                            with open(osp.join(save_dir, 'mapping.pckl'), 'wb') as handle:
                                pickle.dump(dict_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            model = (get_model(model_name))(optimizer=optimizer,
                                                            batch_size=batch_size,
                                                            num_classes=num_classes,
                                                            save_dir=save_dir,
                                                            epochs=epochs)

                            model.train(X_train,
                                        y_train_cate,
                                        X_valid,
                                        y_valid_cate,
                                        weight_class=weight_class,
                                        lr_type=lr_type)

                            current_param = {'optimizer': optimizer,
                                             'learning_rate': lr_type,
                                             'data_type': data_type,
                                             'num_of_classes': num_classes,
                                             # 'class_types': class_type,
                                             'smooth_factor': smooth_factor,
                                             'model': model_name}
                            print(f'Current parameters are {current_param}')

                            y_pred = model.predict(X_test)
                            # reverse_mapping = dict((label, index) for index, label in dict_mapping.items())
                            # real_y = [dict_mapping[y] for y in y_test]
                            # pred_y = [(dict_mapping[y[0]], y[1]) for y in y_pred]
                            # real_y = [reverse_mapping[y] for y in y_test]
                            # pred_y = [(reverse_mapping[y[0]], y[1]) for y in y_pred]

                            y_class_pred = [int(pred[0]) for pred in y_pred]
                            true_conf, true_dicti, false_conf, false_dicti = create_dict_conf(y_test,
                                                                                              y_pred,
                                                                                              num_classes)

                            test_acc = model.model.evaluate(X_test, y_test_cate, batch_size=64)[1]

                            plot_confusion_matrix(y_test, y_class_pred,
                                                  np.array(list(dict_mapping.keys())),
                                                  save_dir,
                                                  normalize=True)

                            plot_confidence(true_conf, false_conf, save_dir, show_plot=False)
                            plot_classes(y_test, y_pred, save_dir, num_classes, show_plot=False)
                            plot_confidence_per_class(true_dicti, false_dicti, num_classes, save_dir, show_plot=False)
                            plot_history(save_dir)

                            if test_acc > acc:
                                acc = test_acc
                                best_parameters = current_param
    return best_parameters, acc


def find_val_acc(weights_dir, name=None):
    file_paths = glob2.glob(osp.join(weights_dir, '**/*.hdf5'))

    val_acc = []
    file_names = []
    for index, file in enumerate(file_paths):
        if name is not None:
            if name in file:
                split_file = file.split("/")
                acc = split_file[-1][-10:-5]
                if file not in file_names:
                    file_names.append(file)
                    val_acc.append(float(acc))
        else:
            split_file = file.split("/")
            acc = split_file[-1][-10:-5]
            if file not in file_names:
                file_names.append(file)
                val_acc.append(float(acc))

    return val_acc, file_names


def fin_top_acc(val_acc, file_names, top_acc):
    val_acc = np.array(val_acc)
    file_names = np.array(file_names)

    indices = val_acc.argsort()
    top_val_acc = val_acc[indices[-top_acc:]]
    top_names_acc = file_names[indices[-top_acc:]]

    return top_val_acc, top_names_acc


if __name__ == '__main__':
    # best_acc = search()
    val_acc, file_names = find_val_acc(NS_WEIGHTS_DIR, name='LSTM')
    top_val_acc, top_names_acc = fin_top_acc(val_acc, file_names, 80)
    print()
