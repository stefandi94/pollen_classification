import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.settings import NUM_OF_CLASSES
from utils.utilites import count_values


def show_values(pc, fmt="%.2f", **kw):
    """
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    """
    from itertools import zip_longest as zip
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    """
   Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    """
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
            correct_orientation=False, cmap='RdBu'):
    """
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    """

    # Plot it out
    fig, ax = plt.subplots()
    # c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    # fig.set_size_inches(cm2inch(40, 20))
    # fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))


def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    """
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    """
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 3)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
            correct_orientation, cmap=cmap)


def plot_pred(report):
    sampleClassificationReport = report
    plot_classification_report(sampleClassificationReport)
    plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()


def plot_confidence(y_true, y_pred, path, num_classes):
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

    print(f'Confidence for true prediction is {np.mean(y_true)}, and for false prediction is {np.mean(y_pred)}')
    bins = np.arange(0, 1, 0.05)
    false_pred, _ = np.histogram(false_conf, bins)
    true_pred, _ = np.histogram(true_conf, bins)

    legend = ['misses', 'hits']
    ax = plt.subplot(111)
    ax.bar(bins[1:] + 0.0325, false_pred, width=0.015, color='b', align='center')
    ax.bar(bins[1:] - 0.0325, true_pred, width=0.015, color='r', align='center')
    plt.legend(legend, loc='best')
    plt.savefig(os.path.join(path,'conf.png'))
    plt.show()

    return true_dicti, false_dicti


def confusion_matrix(save, cm, classes, normalize, klasa, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=((30, 30)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    matplotlib.rcParams.update({'font.size': 20})
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    if save:
        plt.savefig('/home/pedjamat/Documents/models/cm/' + str(klasa) + '.png')
    else:
        plt.show()


def plot_tr(epoch_list, all_train_losses, all_validation_losses):
    plt.figure(figsize=(10, 8))
    plt.plot(epoch_list, all_train_losses)
    plt.plot(epoch_list, all_validation_losses)
    plt.xlabel("Epoch number")
    plt.ylabel("Cost")
    plt.legend(["Train", "Validation"], fontsize=20)
    matplotlib.rcParams.update({'font.size': 20})
    # plt.savefig('/home/pedjamat/Pictures/overfitting.png')
    plt.show()


def plot_classes(y_true, y_pred, path, num_of_classes):
    class_pred = [clas[0] for clas in y_pred]
    pred_dict = count_values(class_pred)
    true_dict = count_values(y_true)

    for i in range(num_of_classes):
        if i not in pred_dict.keys():
            pred_dict[i] = 0

    legend = ['predicted', 'true']
    plt.figure(figsize=(15, 10))
    plt.bar(np.arange(-0.2, num_of_classes - 1, 1), list(pred_dict.values()), width=0.3, align='center', color='r')
    plt.bar(np.arange(0.2, num_of_classes, 1), list(true_dict.values()), width=0.3, align='center', color='b')
    plt.xticks(range(len(true_dict)), list(true_dict.keys()))
    plt.legend(legend, loc='best')
    plt.savefig(os.path.join(path, 'classes.png'))
    plt.show()


def plot_confidence_per_class(true_conf_dict, false_conf_dict, num_class):
    bins = np.arange(0, 1, 0.05)
    true_classes_conf, _ = np.histogram(true_conf_dict[num_class], bins)
    false_conf_dict, _ = np.histogram(false_conf_dict[num_class], bins)

    legend = ['predicted', 'true']
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    ax.bar(bins[1:] + 0.0325, true_classes_conf, width=0.015, color='b', align='center')
    ax.bar(bins[1:] - 0.0325, false_conf_dict, width=0.015, color='r', align='center')
    plt.legend(legend, loc='best')
    plt.savefig(f'./conf_class_{num_class}.png')
    plt.show()
