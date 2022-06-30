# Imports
import torch
import torchvision.transforms as tf

import matplotlib.pyplot as plt
import numpy as np
import statistics
import math


def my_transform(crops):
    return torch.stack([tf.ToTensor()(crop) for crop in crops])


def print_scores(conf_pred, conf_truth):
    tp, fp, tn, fn = confusion(conf_pred, conf_truth)
    print("TN, FP, FN, TP")
    print([tn, fp, fn, tp])
    print(f"Prec:   {tp/(tp+fp):>2.4%}  | TP/(TP+FP)")
    print(f"Recall: {tp/(tp+fn):>2.4%}  | TP/(TP+FN)")



def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)

    Source: https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

# TODO: Save plot somehow with parameters used


def save_plot(iteration, loss, labels, Folder):
    plt.title("Loss graph")
    plt.xlabel('Epochs')
    plt.xticks(np.arange(min(iteration), max(iteration)+2, step=2))
    plt.ylabel('Loss')
    plt.plot(iteration, loss)
    plt.legend(labels)
    plt.savefig(f"{Folder}/loss_graph.png")


def multiple_histo(models_checked: dict, file):
    amount = len(models_checked.keys())
    amount = 10

    fig, axs = plt.subplots(2, amount)

    fig.set_size_inches(60, 15)
    fig.suptitle(f'Validation histograms of multiple cameras')
    plt.rcParams["font.family"] = "monospace"

    bin_size = 0.10
    for e_out, (k, v) in enumerate(list(models_checked.items())[:amount]):

        for e, (name, values) in enumerate([("same", v['same']), ("different", v['diff'])]):
            ax_plot = axs[e, e_out]
            ax_plot.set_title(
                f'Histogram of {name} camera pairs \n(N = {len(values)})')
            try:
                mean, std = statistics.mean(values), statistics.stdev(values)
            except statistics.StatisticsError:
                mean, std = 0, 0
            except TypeError:
                mean, std = 0, 0
                # print(values)
                print("Broken on type error")
            ax_plot.hist(values, bins=np.arange(
                max(0, math.floor(mean-std-bin_size)), math.ceil(mean+std+bin_size), bin_size))

            textstr = f"{k}\n"
            textstr += f"Mean  = {mean:.4f}\n"
            textstr += f"Stdev = {std:.4f}\n"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax_plot.text(0.6, 0.95, textstr, transform=ax_plot.transAxes, fontsize=12,
                         verticalalignment='top', bbox=props)
            ax_plot.set_ylim(top=math.ceil(ax_plot.get_ylim()[1]*1.10))
            ax_plot.set_xlim(right=1)

    for ax in axs.flat:
        ax.set(xlabel='Distance', ylabel='Amount ')

    fig.tight_layout()

    plt.savefig(file)


def summed_histo(summed: dict, file):
    bin_size = 0.05
    fig, axs = plt.subplots(2, 1, sharey='all')
    fig.set_size_inches(20, 15)
    fig.suptitle(f'Validation histograms of all cameras summed together')
    plt.rcParams["font.family"] = "monospace"

    for e, (k, values) in enumerate(summed.items()):
        ax_plot = axs[e]
        ax_plot.set_title(
            f'Histogram of {"different" if k == "diff" else k} camera pairs \n'
            f'(N = {len(values)})')
        try:
            mean, std = statistics.mean(values), statistics.stdev(values)
        except statistics.StatisticsError:
            mean, std = 0, 0
        ax_plot.hist(values, bins=np.arange(
            max(0, math.floor(mean-std-bin_size)), math.ceil(mean+std+bin_size), bin_size))

        textstr = (
            f"Summed\n"
            f"Mean  = {mean:.4f}\n"
            f"Stdev = {std:.4f}\n")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ax_plot.text(0.6, 0.95, textstr, transform=ax_plot.transAxes,
                     fontsize=12, verticalalignment='top', bbox=props)
        ax_plot.set_xlim(right=1, left=0)

    for ax in axs.flat:
        ax.set(xlabel='Distance', ylabel='Amount ')

    fig.tight_layout()
    plt.savefig(file)
