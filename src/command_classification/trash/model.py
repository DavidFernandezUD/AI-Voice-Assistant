import torch
import torch.nn as nn
import torch.functional as F


class CommandModel(nn.Module):

    def __init__(self):
        super(CommandModel, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 32, (3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(44544, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 21)
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.classifier(x)

        return x
