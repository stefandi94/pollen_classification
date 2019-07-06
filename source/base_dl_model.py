import os
import os.path as osp
from abc import abstractmethod
from contextlib import redirect_stdout
from typing import List, Tuple

import keras
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.metrics import top_k_categorical_accuracy
from keras_self_attention import SeqSelfAttention

from settings import NUM_OF_CLASSES
from source.learning_rates.cyclical_lr import CyclicLR
from utils.utilites import multiple_generator


class BaseDLModel:
    cnn_shapes = {
        'input_shape_1': (4, 32, 1),
        'input_shape_2': (4, 24, 1),
    }
    rnn_shapes = {
        'input_shape_3': (20, 120, 1),
        'input_shape_4': (5, 1, 1)
    }

    def __init__(self,
                 epochs: int = 30,
                 batch_size: int = 32,
                 num_classes: int = NUM_OF_CLASSES,
                 **kwargs) -> None:
        """
        :param epochs: num of epochs
        :param batch_size:
        :param num_classes:
        :param optimizer:
        :param kwargs:
        """

        allowed_kwargs = ['save_dir', 'load_dir', 'batch_size', 'epochs', 'optimizer', 'learning_rate']

        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.save_dir = None
        self.load_dir = None

        self.optimizer = None
        self.learning_rate = None

        for k in kwargs.keys():
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed: ' + f'{k}')
            else:
                self.__setattr__(k, kwargs[k])

    @abstractmethod
    def build_model(self) -> None:
        """Builds model"""

        raise NotImplementedError

    def train(self,
              X_train: np.ndarray or List[np.ndarray],
              y_train: np.ndarray or List[np.ndarray],
              X_valid: np.ndarray or List[np.ndarray],
              y_valid: np.ndarray or List[np.ndarray],
              weight_class: np.ndarray,
              generator: bool = False) -> None:
        """
        Train model given parameters
        :param X_train: train data
        :param y_train: train classes
        :param X_valid: validation data
        :param y_valid: validation classes
        :param weight_class: weights for
        :param generator
        :return:
        """

        self.build_model()

        # Given path, creates folders to that path
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f'Created path for model at: {self.save_dir}')

        # If load path is given, load model from that path
        if self.load_dir is not None:
            self.load_model(self.load_dir)
            print(f'Model is loaded from {self.load_dir}')

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.optimizer(self.learning_rate),
                           metrics=['accuracy', top_k_categorical_accuracy, self.precision, self.recall, self.f1])

        weights_name = "{epoch}-{loss:.3f}-{acc:.3f}-{val_loss:.3f}-{val_acc:.3f}.hdf5"

        # print(self.model.summary())
        with open(osp.join(self.save_dir, 'model_summary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
        checkpoint = ModelCheckpoint(os.path.join(self.save_dir, weights_name),
                                     monitor='val_acc',
                                     verbose=1,
                                     save_weights_only=False,
                                     save_best_only=True,
                                     mode='max')

        clr = CyclicLR(max_lr=0.008, mode='exp_range')
        csv_logger = CSVLogger(osp.join(self.save_dir, "model_history_log.csv"), append=True)
        callbacks_list = [checkpoint, csv_logger, clr]

        if not generator:
            self.model.fit(X_train, y_train,
                           validation_data=(X_valid, y_valid),
                           verbose=1,
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           callbacks=callbacks_list)

        else:
            self.model.fit_generator(multiple_generator(X_train, y_train, batch_size=self.batch_size),
                                     steps_per_epoch=X_train[0].shape[0] // self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=multiple_generator(X_valid, y_valid, batch_size=self.batch_size),
                                     validation_steps=X_valid[0].shape[0] // self.batch_size,
                                     use_multiprocessing=True,
                                     callbacks=callbacks_list,
                                     class_weight=weight_class)

    def predict(self, X_test: List[np.ndarray] or np.ndarray) -> List[Tuple[int, float]]:
        """Return prediction for given data."""
        self.load_model(self.load_dir)
        print(f'Model is loaded from {self.load_dir}')

        all_predictions = self.model.predict(X_test)
        predicted_class = np.argmax(all_predictions, axis=1)
        confidence = np.array([all_predictions[vec_num][idx] for vec_num, idx in enumerate(predicted_class)])
        return list(zip(predicted_class, confidence))

    def load_model(self, path: str) -> None:
        """Load model from given path."""

        self.model = keras.models.load_model(path, custom_objects={'SeqSelfAttention': SeqSelfAttention(),
                                                                   'recall': self.recall,
                                                                   'precision': self.precision,
                                                                   'f1': self.f1})

    def save_model(self, path: str) -> None:
        """Save model on given path."""

        try:
            os.makedirs(osp.dirname(path), exist_ok=True)
            self.model.save(path)
        except Exception as e:
            print(e)
            print("Couldn't save model on path {}!".format(path))

    @staticmethod
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
