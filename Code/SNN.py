# Imports


import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.utils.data import Dataset
from torch import Tensor, flatten, clamp

import multiprocessing
import random
from typing import Tuple
from pathlib import Path
from PIL import Image


class SNNDataset(Dataset):
    """Custom dataset for the use of the SiameseNetwork """

    def __init__(self, image_paths: list, transform=None) -> None:
        """Initialization

        Args:
            image_paths (list): 
                List of paths to the images used for this set
            transform (tranforms, optional): 
                Torch tranformation that will be applied to all images. 
                Defaults to None.
        """

        self.image_paths = image_paths
        # image_paths should be in the form [(<path>, <label>), ..., (<path>, <label>)],
        #    where label should be stored somewhere to refer back to the name of the camera i think
        self.transform = transform
        self.printBound = max(100, len(image_paths)//10)

        self.dict = multiprocessing.Manager().dict()
        self.images = multiprocessing.Manager().dict()
        self.leftover = multiprocessing.Manager().list(range(len(image_paths)))
        # Multiprocessing the loading of the images, since it took around 40 minutes to load the whole set
        pool = multiprocessing.Pool()

        pool.map(self.load_images, enumerate(image_paths))
        self.dict = dict(self.dict)
        pool.close()

    def load_images(self, img_obj: Tuple[int, Tuple[Path, int]]) -> None:
        """ Function for preloading the images of the dataset

        Args:
            img_obj (Tuple[int, Tuple[Path, int]]): 
                Enumerated tuple of image path and label
        """

        idx, (path, label) = img_obj

        # Printing of how far the loading is
        try:
            self.leftover.remove(idx)
        except TypeError as e:
            # Sometimes when there are only a few images the multiprocessing can break
            print(idx, type(idx))
            print(e)
        if len(self.leftover) % self.printBound == 0:
            print(f"Images left: {len(self.leftover):>6}", flush=True)

        # Keeping a label -> image idx map for faster selection times
        try:
            self.dict[label].append(idx)
        except KeyError:
            self.dict[label] = [idx]

        # Open image and convert to greyscale and store in RAM
        img0 = Image.open(path).convert("L")
        self.images[path] = (self.transform(img0))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, str, str]:
        """Generates a pair of images (one sample)

        Args:
            index (int): Index of the sample

        Returns:
            Tuple[Tensor, Tensor, Tensor, str, str]: 
                Return a tuple of the image pair, truth label, and file names
        """

        img0_tuple = self.image_paths[index]  # (<path>, <label>)

        # We need to approximately 50% of images to be in the same class
        # Making use of the label -> image idx map created at initialization of the class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            img1_tuple = self.image_paths[random.choice(
                self.dict.get(img0_tuple[1]))]
        else:
            selection = list(self.dict.keys())
            selection.remove(img0_tuple[1])
            key = random.choice(selection)
            img1_tuple = self.image_paths[random.choice(self.dict.get(key))]

        # model/picture.jpg
        f_name0 = "/".join(str(img0_tuple[0]).split("/")[-2:])
        f_name1 = "/".join(str(img1_tuple[0]).split("/")[-2:])

        # Ground truth is a tensor of one value.
        # Could be changed into img0 != img1, however this expression is a bit more clear
        ground_truth = Tensor([0. if img1_tuple[1] == img0_tuple[1] else 1.])

        # Get the image tensors from RAM
        img0 = self.images[img0_tuple[0]]
        img1 = self.images[img1_tuple[0]]

        # Perform Vertical or Horizontal flip
        # Half of the batches will be flipped ->
        #   (50% Not flipped, 25% Horizontal flip, 25% Vertical flip)
        if random.randint(0, 1):
            if random.randint(0, 1):
                rotation = tf.RandomHorizontalFlip(p=1)
                img0 = rotation(img0)
                img1 = rotation(img1)
            else:
                rotation = tf.RandomVerticalFlip(p=1)
                img0 = rotation(img0)
                img1 = rotation(img1)

        return img0, img1, ground_truth, f_name0, f_name1

    def __len__(self) -> int:
        """Denotes the total number of samples

        Returns:
            int: Return the total number of samples
        """

        return len(self.image_paths)


# create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    """ The siamese neural network representation as a class """

    def __init__(self, size: str) -> None:
        """ Initialization

        Args:
            size (str): 
                The size selection for the neural network. 
                This changes the layers used. 
        """

        super(SiameseNetwork, self).__init__()
        # Select the wanted layers via the local function
        getattr(self, size)()

    def forward_once(self, input_image: Tensor) -> Tensor:
        """ A custom function that helps with doing the step for a pair of images

        Args:
            input_image (Tensor): 
                The input image which will be transformed into a feature vector

        Returns:
            Tensor: Return the extracted feature vector
        """

        output = self.CNN(input_image)
        output = output.view(output.size()[0], -1)
        # output = flatten(output, start_dim=1)
        output = self.FC(output)

        return output

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """ The pytorch function that will apply a step in the training

        Args:
            input1 (Tensor): Image one of the pair
            input2 (Tensor): Image two of the pair

        Returns:
            Tensor: Return a tensor containg the distance of the image pair
        """
        # In this function we pass in both images and obtain both vectors
        # which are returned
        bs, ncrops, c, h, w = input1.size()
        output1 = self.forward_once(input1.view(-1, c, h, w))

        output2 = self.forward_once(input2.view(-1, c, h, w))

        # Calculated the similarity through the euclidean distance
        euclidean_distance = F.pairwise_distance(
            output1, output2, keepdim=True).view(bs, ncrops, -1).mean(1)

        return clamp(euclidean_distance, max=1)

    # LAYERS, These should correspond with the option in the argument parser
    def small(self):
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.FC = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128)
        )

    def medium(self):
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 160, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(160, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True)
        )
        self.FC = nn.Sequential(
            nn.Linear(25600, 8192),
            nn.ReLU(inplace=True),

            nn.Linear(8192, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 128)
        )

    def large(self):
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=5, stride=2),
            nn.ReLU(inplace=True)
        )

        self.FC = nn.Sequential(
            nn.Linear(3456, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 1024)
        )

    # Credits: Guru
    def guru(self):
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(
                7, 7), padding=(3, 3), stride=(2, 2)),

            # Block 1
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(in_channels=96, out_channels=64,
                      kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.FC = nn.Sequential(

            nn.Linear(in_features=2048, out_features=1024),
            nn.Tanh(),
            nn.Dropout(p=0.3)
        )
