import matplotlib.pyplot as plt
import numpy as np
from os import path
import statistics
import math
import json
import time
import torch
  
'''Source: https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d'''  
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
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
def save_plot(iteration, loss, Folder):
    plt.title("Loss graph")
    plt.xlabel('Epochs')
    plt.xticks(np.arange(min(iteration), max(iteration)+2, step=2))
    plt.ylabel('Loss')
    plt.plot(iteration, loss)
    plt.savefig(f"{Folder}/loss_graph.png")


def histo_makers(same, diff, validate_name, file):
    fig, axs = plt.subplots(2, 1)
    fig.suptitle(f'Validating against: {validate_name}')
    plt.rcParams["font.family"] = "monospace"
        
    w=0.25

    for e, (name, values) in enumerate([("same", same), ("different" ,diff)]):
        axs[e].set_title(f'Histogram of {name} camera pairs (N = {len(values)})')
        mean, std = statistics.mean(values), statistics.stdev(values)
    
        axs[e].hist(values, bins=np.arange(max(0, math.floor(mean-std-w)), math.ceil(mean+std+w), w))
        # TODO: See if i can make a zoomed version between 0 and 1
        textstr  = f"Mean  = {mean:.4f}\n"
        textstr += f"Stdev = {std:.4f}\n"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        axs[e].text(1.05, 0.95, textstr, transform=axs[e].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        axs[e].set_ylim(top=math.ceil(axs[e].get_ylim()[1]*1.10))
       
        print(f"  For the {name} camera pairs: mean = {mean:.4f} with std = {std:.4f}")
    

    for ax in axs.flat:
        ax.set(xlabel='Distance', ylabel='Amount ')

    fig.tight_layout()
  
    plt.savefig(file)

def multiple_histo(models_checked, file):
    amount = len(models_checked.keys())

    fig, axs = plt.subplots(2, amount)

    fig.set_size_inches(60, 15)
    fig.suptitle(f'Validation histograms of multiple cameras')
    plt.rcParams["font.family"] = "monospace"
        
    w=0.10
    for e_out, (k, v) in enumerate(models_checked.items()):
       
        for e, (name, values) in enumerate([("same", v['same']), ("different" ,v['diff'])]):
            ax_plot = axs[e, e_out]
            ax_plot.set_title(f'Histogram of {name} camera pairs \n(N = {len(values)})')
            try:
                mean, std = statistics.mean(values), statistics.stdev(values)
            except statistics.StatisticsError:
                mean, std = 0, 0
            except TypeError:
                mean, std = 0, 0
                # print(values)
                print("Broken on type error")
            ax_plot.hist(values, bins=np.arange(max(0, math.floor(mean-std-w)), math.ceil(mean+std+w), w))
            
            textstr  = f"{k}\n"
            textstr += f"Mean  = {mean:.4f}\n"
            textstr += f"Stdev = {std:.4f}\n"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax_plot.text(0.6, 0.95, textstr, transform=ax_plot.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            ax_plot.set_ylim(top=math.ceil(ax_plot.get_ylim()[1]*1.10))
            ax_plot.set_xlim(right=1.1)
    

    for ax in axs.flat:
        ax.set(xlabel='Distance', ylabel='Amount ')

    fig.tight_layout()
  
    plt.savefig(file)


# ONLY FOR LOCAL MODE

# Creating some helper functions
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()