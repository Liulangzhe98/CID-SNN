# Used for typing annotation
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configure import Configure
    from SNN import SiameseNetwork

# Imports
from torch.utils.data import DataLoader as DL

from collections import defaultdict
import time

from SNN import *
from helper import *
from splitter import *


def test_model(net: SiameseNetwork, dataloader: DL,
                     config: Configure) -> None:
    """ Function that will test the model

    Args:
        net (SiameseNetwork): The model to be test
        dataloader (DL): The testing dataloader that provides the images in batches
        config (Configure): Configure object containing important attributes
    """
    print(" === Testing results === ")

    net.train(False)
    conf_truth = []
    conf_pred = []
    epsilon = 0.5
    DEVICE = config.DEVICE

    models_checked = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for img0, img1, gt, f1, f2 in dataloader:
            # img0, img1 are list of images
            # print(f"{i:>4} / {len(dataloader.dataset)} | {f1[0].split('/')[0]:30} vs {f2[0].split('/')[0]:30}")

            # Send the images and labels to CUDA
            img0, img1, gt = img0.to(DEVICE), img1.to(DEVICE), gt.to(DEVICE)

            # Pass in the images into the network and obtain the outputs
            # The amount of images passed is equal to the batch size
            prediction = net(img0, img1)
            difference = torch.clamp(prediction, max=0.99999).item()
            conf_truth.append(gt)

            # Could be changed into difference >= epsilon, however this expression is a bit more clear
            pred_label = 0. if difference < epsilon else 1
            conf_pred.append(pred_label)

            basename = "_".join(str(f1[0]).split("_")[:2])
            if gt == 1:
                # The ground truth says they are different
                models_checked[basename]["diff"].append(difference)
            else:
                models_checked[basename]["same"].append(difference)
    models_checked["Summed"] = {
        "same": [x for (_, v) in models_checked.items() for x in v["same"]],
        "diff": [x for (_, v) in models_checked.items() for x in v["diff"]]
    }

    multiple_histo(models_checked, epsilon, config.MULTIPLE)
    summed_histo(models_checked["Summed"], epsilon, config.SUMMED)
    print_results(torch.Tensor(conf_pred), torch.Tensor(conf_truth))

def preload_testing(transformation: tf, test_set: list) -> SNNDataset:
    """Wrapper for the preloading of training/validation images

    Args:
        transformation (torchvision.transformation): 
                Transformations that should be applied to the images
        test_set (list) : 
                The list of image paths for testing

    Returns:
        SNNDataset: 
            Return the SNNDataset object of the testing set
    """

    start = time.time()
    print("Pre loading testing images")
    test_data = SNNDataset(image_paths=test_set, transform=transformation)
    print(f"Loading all images took: {time.time()-start:.4f}s")
    return test_data