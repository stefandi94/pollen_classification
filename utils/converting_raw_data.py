#!/usr/bin/env python
# coding: utf-8
import json
import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from settings import RAW_DATA_DIR, DATA_DIR
from split_data import load_data
from utils.preprocessing import label_to_index, calculate_and_check_shapes


def transform_raw_data(raw_data_path):
    files = sorted(os.listdir(raw_data_path))
    data = [[], [], [], [], []]
    targets = []

    class_to_num = label_to_index(files)

    for file_name in files:
        if file_name.split(".")[-1] != 'json':
            continue

        print(f'Current file_name is {file_name} and time is {datetime.now().time()}')
        raw_data = json.loads(open(osp.join(raw_data_dir, file_name)).read())

        for i in range(len(raw_data["Data"])):
            specmax = np.max(raw_data["Data"][i]["Spectrometer"])
            file_data = raw_data["Data"][i]
            calculate_and_check_shapes(file_data, file_name, specmax, data, targets, class_to_num)

        print(f'Current length of data is: {len(data[0])}')

    feature_names = ['scatter', 'size', 'life_1', 'spectrum', 'life_2']

    files_to_save = [data, targets, class_to_num, feature_names]
    filenames_to_save = ['data.pckl', 'labels.pckl', 'label_to_index.pckl', 'feature_names.pckl']

    for index, filename_to_save in enumerate(filenames_to_save):
        f = open(osp.join(DATA_DIR, f'{filename_to_save}'), 'wb')
        pickle.dump(files_to_save[index], f)
        f.close()


def create_lifetime(data):
    lista = []
    for i in range(len(data[2])):
        if i == 0:
            l = ["Cupressus"]
        elif i == 1:
            l = ["Fraxinus excelsior"]
        else:
            l = ["Ulmus"]

        if np.where(data[2][i][0, :] == np.max(data[2][i][0, :]))[0].shape[0] > 1:
            l.append("Yes")
        else:
            l.append("No")

        for k in range(4):
            if k != 2:
                l.append(np.max(data[2][i][k, :]) / np.e)
        lista.append(l)

    features = ["Pollen type", "Saturated", "Time of band 1", "Time of band 2", "Time of band 3"]
    amb = pd.DataFrame(columns=features)

    for i in range(len(lista)):
        print(f'Current index is {i}')
        amb.loc[len(amb)] = lista[i]

    amb.to_csv("./../data/Time of lifetime.csv", index=False)


if __name__ == '__main__':
    raw_data_dir = '/mnt/hdd/data/'
    # transform_raw_data(raw_data_dir)
    data = load_data(raw_data_dir, 'data')
    create_lifetime(data)
