from torch.utils.data import Dataset
import random
from PIL import Image
import torch
import numpy as np
import torch.nn as nn

class DresdenSNNDataset(Dataset):
    def __init__(self, image_paths: list,transform=None):
        self.image_paths = image_paths  
        # image_paths should be in the form [(<path>, <label>), ..., (<path>, <label>)],
        #    where label should be stored somewhere to refer back to the name of the camera i think
        self.transform = transform
        
    def __getitem__(self,index):
        if torch.is_tensor(index):
            idx = idx.tolist()
        
        img0_tuple = random.choice(self.image_paths) # (<path>, <label>)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.image_paths) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.image_paths)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        # Convert to greyscale
        img0 = img0.convert("L") 
        img1 = img1.convert("L")

        f_name0 = "/".join(str(img0_tuple[0]).split("/")[-2:]) # model/picture.jpg
        f_name1 = "/".join(str(img1_tuple[0]).split("/")[-2:])
        
        same_label = torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return img0, img1, same_label, f_name0, f_name1

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
            nn.Linear(10752, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
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
        return output1, output2


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