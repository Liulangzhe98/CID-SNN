from torch.utils.data import Dataset
import random
from PIL import Image
import torch
import numpy as np
import torch.nn as nn

from collections import defaultdict

class DresdenSNNDataset(Dataset):
    def __init__(self, image_paths: list,transform=None):
        self.image_paths = image_paths  
        # image_paths should be in the form [(<path>, <label>), ..., (<path>, <label>)],
        #    where label should be stored somewhere to refer back to the name of the camera i think
        self.transform = transform

        self.dict = defaultdict(list)
        self.images = {}
        
        for e, (x, y) in enumerate(image_paths):
            self.dict[y].append(e)
            # Open image and convert to greyscale
            img0 = Image.open(x).convert("L")
            self.images[x] = (self.transform(img0))
        
    def __getitem__(self, index):
        img0_tuple = self.image_paths[index] # (<path>, <label>)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            img1_tuple = self.image_paths[random.choice(self.dict.get(img0_tuple[1]))]
        else:
            key = random.choice(list(self.dict.keys()-[img0_tuple[1]]))
            img1_tuple = self.image_paths[random.choice(self.dict.get(key))]

        f_name0 = "/".join(str(img0_tuple[0]).split("/")[-2:]) # model/picture.jpg
        f_name1 = "/".join(str(img1_tuple[0]).split("/")[-2:])

        # Label is a tensor of 1 value: 0 when the same
        different = torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
     
        img0 = self.images[img0_tuple[0]]
        img1 = self.images[img1_tuple[0]]

        return img0, img1, different, f_name0, f_name1

    def __len__(self):
        return len(self.image_paths)
   
#create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        # TODO: Set up the layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,128)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        print("small me")
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        x = torch.concat(tensors=(output1, output2), dim=1)
        x = self.fc2(x)
        x = torch.sigmoid(x)
     
        return x



#create the Siamese Neural Network
class SiameseNetworkLarger(nn.Module):
    def __init__(self):
        super(SiameseNetworkLarger, self).__init__()

        # Setting up the Sequential of CNN Layers
        # TODO: Set up the layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=20,stride=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(9600, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,128)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        x = torch.concat(tensors=(output1, output2), dim=1)
        x = self.fc2(x)
        x = torch.sigmoid(x)
     
        return x


# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
      return loss_contrastive