"""
Wakeword detection model.
"""


import torch.nn as nn


class WakewordModel(nn.Module):

    def __init__(self):
        super(WakewordModel, self).__init__()

        # Normilization layer
        self.norm = nn.BatchNorm2d(1)

        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        
        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        # Linear classifier
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(17920, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear(x)

        return x
