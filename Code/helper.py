# Used for typing annotation
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from configure import Configure
    from SNN import SiameseNetwork

# Imports
import torch

import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
import json


def create_experiment_object(config: Configure,  model: SiameseNetwork) -> dict:
    """Function that returns the wanted information about the experiment 

    Args:
        config (Configure): Configure object containing important attributes.
        model (SiameseNetwork): Model used for the experiments.

    Returns:
        dict: Return a dictionary containing information about the experiment. 
    """

    return {
        "Model": config.SAVE_MODEL_AS,
        "UID": config.LOAD_ID,
        "Transform": str(config.Transform).split("\n"),
        "Layers": str(model).split("\n"),
        "Params": config.HYPER,
    }


def save_model(cfg: Configure, SNN_model: SiameseNetwork) -> None:
    """Function that save the experiment data to the json file

    Args:
        cfg (Configure): Configure object containing important attributes
        SNN_model (SiameseNetwork): Model to be saved
    """

    torch.save(SNN_model, cfg.SAVE_MODEL_AS)
    print(f"Saving the model at: {cfg.SAVE_MODEL_AS}")
    with open(cfg.SAVED_EXPERIMENTS, "r") as se:
        try:
            old_experiments = json.load(se)
        except json.JSONDecodeError:
            old_experiments = dict()
        old_experiments[cfg.EXP_ID] = create_experiment_object(cfg, SNN_model)
    with open(cfg.SAVED_EXPERIMENTS, "w") as se:
        json.dump(old_experiments, se, indent=2)


def find_threshold(summed: dict) -> None:
    """Finds the best threshold value used for binary class prediction

    Args:
        same (list): _description_
        diff (list): _description_
    """
    same = summed['same']
    diff = summed['diff']

    highest_f = (0, 0)
    highest_p = (0, 0)
    highest_r = (0, 0)
    for e in range(10, 100):
        e /= 100
        tp = len([x for x in same if x < e])
        fn = len(same) - tp
        fp = len([x for x in diff if x < e])
        tn = len(diff) - fp
        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        fscore = 2*(recall*prec)/(recall+prec)

        if fscore > highest_f[0]:
            highest_f = (fscore, e)
        if prec > highest_p[0]:
            highest_p = (prec, e)
        if recall > highest_r[0]:
            highest_r = (recall, e)

    print(f"The highest precision: {highest_p}")
    print(f"The highest recall   : {highest_r}")
    print(f"The highest fscore   : {highest_f}")


def print_results(conf_pred: torch.Tensor, conf_truth: torch.Tensor) -> None:
    """ Prints out the results of the experiment

    Args:
        conf_pred (torch.Tensor): Tensor of the predictions.
        conf_truth (torch.Tensor): Tensor of the ground truh.
    """
    tp, fp, tn, fn = confusion(conf_pred, conf_truth)

    prec = tp/(tp+fp)
    recall = tp/(tp+fn)

    print("TP, FN, FP, TN")
    print([tp, fn, fp, tn])
    print(f"Acc   : {(tp+tn)/(tp+fp+fn+tn):>7.4%}  | (TP+TN)/(TP+FP+FN+TN)")
    print(f"Prec  : {prec:>7.4%}  | TP/(TP+FP)")
    print(f"Recall: {recall:>7.4%}  | TP/(TP+FN)")
    print(f"Fscore: {2*(recall*prec)/(recall+prec):>7.4%}")


def confusion(prediction: torch.Tensor, truth: torch.Tensor) -> Tuple[int, int, int, int]:
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 0 and 0 (True Positve)
    - 0 and 1 (False Positve)
    - 1 and 1 (True Negative)
    - 1 and 0 (False Negative)

    Freely changed from:
        - https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   nan   where prediction and truth are 0 (True Positive)
    #   0     where prediction is 0 and truth is 1 (False Positive)
    #   1     where prediction and truth are 1 (True Negative)
    #   inf   where prediction is 1 and truth is 0 (False Negative)

    true_positives = torch.sum(torch.isnan(confusion_vector)).item()
    false_positives = torch.sum(confusion_vector == 0).item()
    true_negatives = torch.sum(confusion_vector == 1).item()
    false_negatives = torch.sum(confusion_vector == float("inf")).item()

    return true_positives, false_positives, true_negatives, false_negatives


def save_loss_graph(data: list, labels: list, file_path: str):    
    """Creates and saves the loss graph made during the training of the model

    Args:
        data (list): Data points of the loss scores
        labels (list): Labels used for the graphs
        file_path (str): Path to the file
    """
    plt.title("Loss graph")
    plt.xlabel("Epochs")
    plt.xticks(np.arange(1, len(data)+2, step=2))
    plt.ylabel("Loss")
    plt.plot(range(1, len(data)+1), data)
    plt.legend(*labels)
    plt.savefig(file_path)


def multiple_histo(models_checked: dict, EPS:float, file_path: str):
    """Creates and saves the similarity histograms of multiple cameras

    Args:
        models_checked (dict): Similartiry scores grouped by camera model
        EPS (float): Threshold value used for binary class prediction
        file_path (str): Path to the file
    """    
    amount = len(models_checked.keys())
    amount = 10

    fig, axs = plt.subplots(amount, 2)

    fig.set_size_inches(15, 60)
    fig.suptitle(f"Validation histograms of multiple cameras")
    plt.rcParams["font.family"] = "monospace"

    bin_size = 0.10

    # The histograms are stacked vertically, so they can be presented as an appendix
    for e_out, (k, v) in enumerate(list(models_checked.items())[:amount]):

        for e, (name, values) in enumerate([("same", v["same"]), ("different", v["diff"])]):
            ax_plot = axs[e_out, e]
            ax_plot.set_title(
                f"Histogram of {name} camera pairs \n(N = {len(values)})")
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
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

            ax_plot.text(0.6, 0.95, textstr, transform=ax_plot.transAxes, fontsize=12,
                         verticalalignment="top", bbox=props)
            ax_plot.set_ylim(top=math.ceil(ax_plot.get_ylim()[1]*1.10))
            ax_plot.set_xlim(right=1)
            ax_plot.axvline(x=EPS, color="red")

    for ax in axs.flat:
        ax.set(xlabel="Distance", ylabel="Amount ")

    fig.tight_layout()

    plt.savefig(file_path)


def summed_histo(summed: dict, EPS: float, file_path: str):
    """Creates and saves the similarity histograms summed together

    Args:
        models_checked (dict): Similartiry scores grouped by camera model
        EPS (float): Threshold value used for binary class prediction
        file_path (str): Path to the file
    """    
    bin_size = 0.05
    fig, axs = plt.subplots(2, 1, sharey="all")
    fig.set_size_inches(20, 15)
    fig.suptitle(f"Validation histograms of all cameras summed together")
    plt.rcParams["font.family"] = "monospace"

    # Summed histograms are divided into same and different camera pairs
    # Color scheme is to visualize the confusion matrix
    for e, (k, values) in enumerate(summed.items()):
        ax_plot = axs[e]
        ax_plot.set_title(
            f"Histogram of {'different' if k == 'diff' else k} camera pairs \n"
            f"(N = {len(values)})")

        ax_plot.axvspan(EPS*e, min(1, EPS+e),
                        facecolor="lightgreen", alpha=0.5)
        ax_plot.axvspan(EPS*(1-e), max(EPS, 1-e),
                        facecolor="lightcoral", alpha=0.5)

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
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.75)

        ax_plot.text(0.6, 0.95, textstr, transform=ax_plot.transAxes,
                     fontsize=12, verticalalignment="top", bbox=props)
        ax_plot.set_xlim(right=1, left=0)

        ax_plot.axvline(x=EPS, color="red")

    for ax in axs.flat:
        ax.set(xlabel="Distance", ylabel="Amount ")

    fig.tight_layout()
    plt.savefig(file_path)
