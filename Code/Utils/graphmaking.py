import torch
from torch import nn
from torchviz import make_dot

model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2)),

            # Block 1
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

x = torch.randn(1,1, 800, 800)

g = make_dot(model(x), params=dict(model.named_parameters()))
g.view()
