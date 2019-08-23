import numpy as np
import pandas as pd
from eli5.sklearn import PermutationImportance as PI
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score
from xgboost.sklearn import XGBClassifier


class BaseModel(object):
    """
    Abstract class for all models
    """

    def train(self, X_train, y_train):
        """
        Train a specific model on the training data
        :param X_train:
        :param y_train:
        :return:
        """
        self.model.fit(X_train, y_train)

    def predict(self, inputs):
        """
        Predict classes based on pre-trained model
        :param inputs:
        :return:
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs):
        """
        Return predicted probability of each input based on pre-trained model
        :param inputs:
        :return:
        """
        return self.model.predict_proba(inputs)

    def get_feature_importance(self, features, nb_features=10):
        """
        Get top N most important fetures
        :param features:
        :param nb_features:
        :return:
        """
        idxs = np.where(self.model.feature_importances_ != 0)[0]
        pred_columns = features.columns[idxs]
        feat_importances = pd.Series(self.model.feature_importances_[idxs], index=pred_columns)
        return feat_importances.nlargest(nb_features)

    def get_metrics(self, x, y):
        """
        Get Accuracy, F1 and precision scores for pre-trained model
        :param x:
        :param y:
        :return:
        """
        metrics = dict()
        y_predicted = self.predict(x)
        metrics['Accuracy'] = accuracy_score(y, y_predicted)
        metrics['F1'] = f1_score(y, y_predicted)
        metrics['Precision'] = precision_score(y, y_predicted)

        return metrics

    def load_model(self, path):
        """
        Load model from file system
        :param path:
        :return:
        """
        try:
            self.model = joblib.load(path)
        except Exception as e:
            print(e)
            print("Couldn't load scikit learn model on path {}!".format(path))

    def save_model(self, path):
        """
        Save model to the file system
        :param path: 
        :return: 
        """
        try:
            # os.makedirs(osp.dirname(path), exist_ok=1)
            joblib.dump(self.model, path)
        except Exception as e:
            print(e)
            print("Couldn't save scikit learn model on path {}!".format(path))


class RandomForest(BaseModel):
    """
    Random Forest model wrapper
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=4, max_depth=3, random_state=42)
        self.name = 'RF'


class ExtraTrees(BaseModel):
    """
    Extra trees model wrapper
    """

    def __init__(self):
        self.model = ExtraTreesClassifier(n_estimators=100, n_jobs=4, max_depth=3, random_state=42)
        self.name = 'ET'


class XGBoost(BaseModel):
    """
    XGBoost trees model wrapper
    """

    def __init__(self):
        self.model = XGBClassifier(max_depth=3, n_jobs=4, random_state=42)
        self.name = 'XGB'


class PermutationImportance(BaseModel):
    """
    Permutation importance model wrapper
    """

    def __init__(self, clf):
        self.model = PI(clf, random_state=1)
