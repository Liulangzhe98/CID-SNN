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
    plt.text(100, 8, f'Dissimilarity: {distance:>5.2f}', style='italic',fontweight='bold',
        bbox={'facecolor':'white', 'alpha':1.0, 'pad':20})
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{RESULT_FOLDER}/result_{idx}.png')
    # plt.show() 


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