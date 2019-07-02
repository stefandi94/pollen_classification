import os
import os.path as osp
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def find_best_acc(dir_path):
    files = os.listdir(dir_path)

    splited = []
    val_acc = []
    for file in files:
        splited_file = file.split("-")
        val_acc.append(float(splited_file[3][8:12]))
    max_val_acc_idx = np.argmax(np.array(val_acc))

    best_acc = osp.join(dir_path, files[max_val_acc_idx])
    return best_acc


def count_values(array):
    """
    Given array return dictionary with class numbers and number of instances of that class
    :param array:
    :return:
    """
    unique, counts = np.unique(array, return_counts=True)
    return dict(zip(unique, counts))


def force_class(X, y, limit):
    dicti = count_values(y)

    new_X, new_y = [], []
    for i in range(len(y)):
        if dicti[y[i]] < limit:
            new_X.append(X[i])
            new_y.append(y[i])

    return np.array(new_X), np.array(new_y)


def load_data(path=os.path.join(JOINED_IMAGES, 'data.pckl')):
    """
    :param path: path to dataset
    :return: X_train, y_train, X_valid, y_valid, X_test, y_test
    """
    with open(path, 'rb') as fp:
        images = pickle.load(fp)

    images = shuffle(images, random_state=RANDOM_STATE)

    X = []
    y = []

    for image in images:
        X.append(image[0])
        y.append(image[1])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=0.1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=RANDOM_STATE, test_size=0.1)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def create_train_valid_test(path, name, save_target=False):

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(path)

    if not os.path.exists(TRAINING_DIR):
        os.mkdir(TRAINING_DIR)
        print('Training directory created!')

    if not os.path.exists(VALID_DIR):
        os.mkdir(VALID_DIR)
        print('Valid directory created!')

    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
        print('Test directory created!')

    with open(osp.join(TRAINING_DIR, f'{name}.pckl'), 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(osp.join(VALID_DIR, f'{name}.pckl'), 'wb') as handle:
        pickle.dump(X_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(osp.join(TEST_DIR, f'{name}.pckl'), 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if save_target:
        with open(osp.join(TRAINING_DIR, 'target.pckl'), 'wb') as handle:
            pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(osp.join(VALID_DIR, 'target.pckl'), 'wb') as handle:
            pickle.dump(y_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(osp.join(TEST_DIR, 'target.pckl'), 'wb') as handle:
            pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


def change_float_type(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)

    if isinstance(data, list):
        data = np.array(data)

    data = data.astype('float32')
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

    del data


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_data(path_to_load, path_to_save, file_name, num_files):
    if not osp.exists(path_to_save):
        os.makedirs(path_to_save)

    data_path = osp.join(path_to_load, f"{file_name}.pckl")
    # target_path = osp.join(path_to_load, 'target.pckl')

    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    # with open(target_path, 'rb') as fp:
    #     target = pickle.load(fp)

    data_parts = list(divide_chunks(data, num_files))
    # target_parts = list(divide_chunks(target, num_of_files))

    for index in range(len(data_parts)):
        with open(os.path.join(path_to_save, f'{file_name}_{index}.pckl'), 'wb') as handle:
            pickle.dump(data_parts[index], handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(os.path.join(path, 'target_{index}.pckl'), 'wb') as handle:
        #     pickle.dump(target_parts[index], handle, protocol=pickle.HIGHEST_PROTOCOL)

