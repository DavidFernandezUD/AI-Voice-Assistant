"""
Voice command recognition model.
"""


import torch
import torch.nn as nn
import torch.functional as F


class CommandModel(nn.Module):

    def __init__(self):
        super(CommandModel, self).__init__()

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

        # Linear classifier
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(42240, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 35)
        )

    def forward(self, x):

        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":

    from dataset import SpeechCommandsDataset

    DATASET_CSV = "data/speech_commands_v0.02/dataset.csv"
    NOISE_CSV = "data/speech_commands_v0.02/background_noise.csv"

    dataset = SpeechCommandsDataset(DATASET_CSV, NOISE_CSV, length_ms=2000)
    model = CommandModel()

    x, y = dataset[0]
    x = torch.unsqueeze(x, 0)

    pred = model(x)

    print(pred)
