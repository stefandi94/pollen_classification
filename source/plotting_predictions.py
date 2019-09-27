import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from utils.utilites import count_values, smooth_curve


# def plot_confusion_matrix(confusion_matrix, classes, path, show_plot):
#     df_cm = pd.DataFrame(confusion_matrix, index=[i for i in classes], columns=[i for i in classes])
#     plt.figure(figsize=(20, 15))
#     sns.heatmap(df_cm, annot=True)
#     plt.savefig(os.path.join(path, 'conf_matrix.png'))
#
#     if show_plot:
#         plt.show()
def plot_confusion_matrix(y_true, y_pred,
                          classes, path,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          show_plot=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    fig, ax = plt.subplots(figsize=(20, 15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(path, 'conf_matrix.png'))

    if show_plot:
        plt.show()
    return ax


# np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()


# TODO: create function which will read history log and plot accuracy and loss over time


def create_dict_conf(y_true, y_pred, num_classes):
    true_conf = []
    false_conf = []
    true_dicti = dict((k, []) for k in range(num_classes))
    false_dicti = dict((k, []) for k in range(num_classes))

    for idx, (pred, conf) in enumerate(y_pred):
        if pred == y_true[idx]:
            true_conf.append(conf)
            true_dicti[pred].append(conf)
        else:
            false_conf.append(conf)
            false_dicti[pred].append(conf)

    return true_conf, true_dicti, false_conf, false_dicti


def plot_confidence(true_conf, false_conf, path, show_plot):
    plt.clf()

    bins = np.arange(0, 1, 0.05)
    false_pred, _ = np.histogram(false_conf, bins)
    true_pred, _ = np.histogram(true_conf, bins)

    legend = ['misses', 'hits']
    plt.figure(figsize=(20, 15))
    ax = plt.subplot(111)
    ax.bar(bins[1:] + 0.0325, false_pred, width=0.015, color='b', align='center')
    ax.bar(bins[1:] - 0.0325, true_pred, width=0.015, color='r', align='center')

    plt.legend(legend, loc='best')
    plt.title('Confidence of hits and misses')
    plt.savefig(os.path.join(path, 'conf.png'))

    if show_plot:
        plt.show()


def plot_classes(y_true, y_pred, path, num_of_classes, show_plot):
    plt.clf()

    class_pred = [clas[0] for clas in y_pred]
    pred_dict = count_values(class_pred)
    true_dict = count_values(y_true)

    for i in range(num_of_classes):
        if i not in pred_dict.keys():
            pred_dict[i] = 0

        if i not in true_dict.keys():
            true_dict[i] = 0

    legend = ['predicted', 'true']
    plt.figure(figsize=(20, 15))
    plt.bar(np.arange(-0.2, num_of_classes - 1, 1), list(pred_dict.values()), width=0.3, align='center', color='r')
    plt.bar(np.arange(0.2, num_of_classes, 1), list(true_dict.values()), width=0.3, align='center', color='b')

    plt.xticks(range(len(true_dict)), list(true_dict.keys()))
    plt.legend(legend, loc='best')
    plt.title('Number of true and predicted class')
    plt.savefig(os.path.join(path, 'classes.png'))

    if show_plot:
        plt.show()


def plot_confidence_per_class(true_dicti, false_dicti, num_of_classes, path, show_plot):
    plt.clf()

    bins = np.arange(0, 1, 0.05)
    # uzeti sve klase koje su postoje u true conf i onda naci sve, uzeti mean i plotovati
    mean_true_conf = dict()
    for key in true_dicti:
        mean_true_conf[key] = np.mean(true_dicti[key])

    mean_false_conf = dict()
    for key in true_dicti:
        mean_false_conf[key] = np.mean(false_dicti[key])

    legend = ['true_confidence', 'false_confidence']
    plt.figure(figsize=(20, 15))
    plt.bar(np.arange(-0.2, num_of_classes - 1, 1), list(mean_true_conf.values()), width=0.3, align='center', color='r')
    plt.bar(np.arange(0.2, num_of_classes, 1), list(mean_false_conf.values()), width=0.3, align='center', color='b')

    plt.xticks(range(len(mean_true_conf)), list(mean_false_conf.keys()))
    plt.title('Confidence per class')
    plt.legend(legend, loc='best')
    plt.savefig(os.path.join(path, 'confidence_per_class.png'))

    if show_plot:
        plt.show()


def plot_history(log_path, smooth=False, factor=0.8, show_plot=False):
    plt.clf()
    data = pd.read_csv(os.path.join(log_path, 'model_history_log.csv'))

    plt.clf()
    plt.figure(figsize=(20, 15))
    epochs = range(1, len(data['epochs']))

    acc = data['acc']
    val_acc = data['val_acc']
    loss = data['loss']
    val_loss = data['val_loss']

    if smooth:
        acc = smooth_curve(acc, factor=factor)
        val_acc = smooth_curve(val_acc, factor=factor)
        loss = smooth_curve(loss, factor=factor)
        val_loss = smooth_curve(val_loss, factor=factor)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig(os.path.join(log_path, 'train_valid_acc.png'))

    if show_plot:
        plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(log_path, 'train_valid_loss.png'))

    if show_plot:
        plt.show()
