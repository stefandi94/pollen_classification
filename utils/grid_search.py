import os.path as osp

import glob2
import numpy as np
from keras.utils import to_categorical

from settings import WEIGHTS_DIR
from source.data_loader import data
from source.load_file import get_model
from source.plotting_predictions import plot_confidence, plot_classes
from utils.split_data import save_data
from utils.utilites import smooth_labels


def search():
    best_parameters = {}
    acc = 0
    epochs = 100
    batch_size = 512
    grid = {'data_type': ['normalized', 'standardized'],  # standardized
            'class_types': ['top'],  # down afterwords
            'num_of_classes': [8],
            'smooth_factor': [0, 0.1],
            'optimizer': ['adam', 'rmsprop'],
            'learning_rate': ['cosine'],
            'models': ['ANN', 'LSTM', 'GRU', 'BiLSTM']}
    # 'models': ['ANN', 'CNN', 'CNNRNN' 'RNNLSTM', 'GRU', 'BiLSTM', 'CNNLSTM']}

    for data_type in grid['data_type']:
        for class_type in grid['class_types']:
            for num_classes in grid['num_of_classes']:
                X_train, y_train, X_valid, y_valid, X_test, y_test, weight_class, dict_mapping = data(
                    standardized=data_type,
                    num_of_classes=num_classes,
                    ns=False)

                y_train_cate = to_categorical(y_train, num_classes)
                y_valid_cate = to_categorical(y_valid, num_classes)
                y_test_cate = to_categorical(y_test, num_classes)

                for smooth_factor in grid['smooth_factor']:
                    smooth_labels(y_train_cate, smooth_factor)

                    for optimizer in grid['optimizer']:
                        for lr_type in grid['learning_rate']:
                            for model_name in grid['models']:
                                print('Stared new fitting')
                                save_dir = f'/mnt/hdd/pollen_data/model_weights/os/data_type_{data_type}/' \
                                           f'/smooth_factor_{smooth_factor}' \
                                           f'/optimizer_{optimizer}' \
                                           f'/learning_rate_type_{lr_type}/' \
                                           f'model_name_{model_name}'
                                save_data(dict_mapping, data_path=save_dir, filename='mapping')

                                # save_dir = f'/mnt/hdd/pollen_data/model_weights/os/{data_type}/{model_name}/' \
                                #     f'{class_type}/smooth_factor_{smooth_factor}/{optimizer}/lr_{lr_type}/{num_classes}'
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
                                                 'class_types': class_type,
                                                 'smooth_factor': smooth_factor,
                                                 'model': model_name}
                                print(f'Current parameters are {current_param}')

                                y_pred = model.predict(X_valid)
                                test_acc = model.model.evaluate(X_test, y_test_cate, batch_size=64)[1]

                                plot_confidence(y_valid, y_pred, save_dir, num_classes)
                                plot_classes(y_valid, y_pred, save_dir, num_classes)

                                if test_acc > acc:
                                    acc = test_acc
                                    best_parameters = current_param
    return best_parameters, acc


def find_val_acc():
    weights_dir = WEIGHTS_DIR
    file_paths = glob2.glob(osp.join(weights_dir, '**/*.hdf5'))

    val_acc = []
    file_names = []
    for index, file in enumerate(file_paths):
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
    val_acc, file_names = find_val_acc()
    top_val_acc, top_names_acc = fin_top_acc(val_acc, file_names, 100)
    print()
