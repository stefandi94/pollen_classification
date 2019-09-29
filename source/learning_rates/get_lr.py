import math

import keras

from source.learning_rates.cosine_warmup_lr import WarmUpCosineDecayScheduler
from source.learning_rates.cyclical_lr import CyclicLR


def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.02
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


def choose_lr(type, X_train, batch_size, epochs):

    if type == 'cyclic':
        rate = CyclicLR(base_lr=0.001,
                        max_lr=0.01,
                        mode='triangular',
                        step_size=len(X_train[0]) // (2 * batch_size))

    elif type == 'cosine':
        learning_rate_base = 0.001
        warmup_epoch = int(epochs * 0.2)
        total_steps = int(epochs * len(X_train[0]) / batch_size)
        warmup_steps = int(warmup_epoch * len(X_train[0]) / batch_size)

        # Create the Learning rate scheduler.
        rate = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                          total_steps=total_steps,
                                          warmup_learning_rate=0.0,
                                          warmup_steps=warmup_steps,
                                          hold_base_rate_steps=0)
    return rate
