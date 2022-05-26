import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from os import path


RESULT_FOLDER = "Results"
if not path.exists(RESULT_FOLDER):
    RESULT_FOLDER = "Project/Results"

def save_validation_pairs(img_base, img_val, distance, idx=0):
    # Concatenate the two images together
    concatenated = torch.cat((img_base, img_val), 0)
    torchvision.utils.make_grid(concatenated)

    npimg = torchvision.utils.make_grid(concatenated)
    plt.axis("off")
    plt.text(80, 8, f'Dissimilarity: {distance:>5.2f}', style='italic',fontweight='bold',
        bbox={'facecolor':'white', 'alpha':1.0, 'pad':20})
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{RESULT_FOLDER}/result_{idx}.png')
    # plt.show() 


# TODO: Save plot somehow with parameters used
def save_plot(iteration, loss):
    plt.title("Loss graph")
    plt.xlabel('Batches')
    plt.xlim(left=0)
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.plot(iteration, loss)
    plt.savefig(f"{RESULT_FOLDER}/loss_graph.png")


def histo_makers(same, diff, validate_name):
    fig, axs = plt.subplots(2, 1)
    fig.suptitle(f'Validating against: {validate_name}')

    w=0.25


    axs[0].set_title('Histogram of same camera pairs')
    axs[0].hist(same, density=True, bins=np.arange(min(same), max(same) + w, w))


    axs[1].set_title('Histogram of different camera pairs')
    axs[1].hist(diff, density=True, bins=np.arange(min(diff), max(diff) + w, w))
    

    for ax in axs.flat:
        ax.set(xlabel='Distance', ylabel='Amount (Normalized)')

    fig.tight_layout()
    plt.savefig("Results/histo_together.png")

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