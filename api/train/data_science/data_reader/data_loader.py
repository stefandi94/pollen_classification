import re

import pandas as pd
from sklearn.model_selection import train_test_split
from train.data_science.data_reader.consts import STRING_COLUMNS, CATEGORICAL_COLUMNS, DD2_COLUMNS

regex = re.compile(r"\[|\]|<", re.IGNORECASE)


class DataLoader(object):
    def __init__(self, df=None):
        self.data = df
        self.data = self.data[DD2_COLUMNS]
        self.data = self.data.drop(STRING_COLUMNS, axis=1)
        self.data.fillna('', inplace=True)

    def clean_data(self, target):
        data_num = self.data.drop(CATEGORICAL_COLUMNS, axis=1)
        # data_cat = self.data.drop(data_num.columns, axis=1)
        data_cat = self.data[CATEGORICAL_COLUMNS]
        data_cat = data_cat.astype('category')
        for s in data_cat:
            data_cat[s] = data_cat[s].cat.codes
        self.data = pd.concat([data_num, data_cat], axis=1)
        self.data.drop(target, axis=1, inplace=True, errors='ignore')

        # self.data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col
        #                    in self.data.columns.values]
        # print(self.data.columns.values)
        return self.data

    def label_data(self, target, target_thresh):
        """
        This could be in transform data as well
        :param target:
        :param target_thresh:
        :return:
        """
        data_num = self.data.drop(CATEGORICAL_COLUMNS, axis=1)
        data_cat = self.data[CATEGORICAL_COLUMNS]
        data_cat = data_cat.astype('category')
        for s in data_cat:
            data_cat[s] = data_cat[s].cat.codes
        self.data = pd.concat([data_num, data_cat], axis=1)

        mask = self.data[target].astype('int32') < target_thresh
        self.data.loc[mask, target] = 1
        mask = self.data[target].astype('int32') >= target_thresh
        self.data.loc[mask, target] = 0

        # self.data.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col
        #                    in self.data.columns.values]
        # print(self.data.columns.values)

    def split_data(self, test_size, target):
        """
        Split data to training/test
        :param test_size:
        :param target:
        :return:
        """
        # sklearn train_test_split use random split
        X, Y = self.data.drop(target, axis=1), self.data[target]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                            random_state=42)
        return x_train, x_test, y_train, y_test

    def transform_data(self):
        """
        select attributes, transform categorical values
        :return:
        """
        pass
