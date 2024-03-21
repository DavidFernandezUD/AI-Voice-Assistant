"""
Load dataset for command recognition model.
"""


import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple
import random
import csv


class SpeechCommandsDataset(Dataset):

    def __init__(self, data_csv: str, noise_csv: str = None, sample_rate: int = 8000, augment: bool = True):
        
        self.sample_rate = sample_rate
        self.augment = augment

        # Load data paths
        self.data = list()
        with open(data_csv, "r") as file:
            reader = csv.reader(file)
            
            for row in reader:
                self.data.append(row)

        # Load noise file paths
        self.noise = None
        if noise_csv is not None:
            self.noise = list()
            with open(noise_csv, "r") as file:
                reader = csv.reader(file)

                for row in reader:
                    self.noise.append(row[0])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:

        file, label = self.data[index]

        waveform, freq = torchaudio.load(file, normalize=True) # NOTE: normalize=True converts INT audio formats into float32

        # Add background noise if noise available and at 50% chance
        if self.noise is not None and random.random() < 0.5:
            noise_rate = random.random() * 0.1
            noise_wave, _ = torchaudio.load(random.choice(self.noise), normalize=True)
            noise_wave = noise_wave[:, :waveform.shape[1]] * noise_rate
            
            waveform  += noise_wave

        waveform = T.Resample(orig_freq=freq, new_freq=self.sample_rate)(waveform)

        torchaudio.save(f"data/test/{index}.wav", waveform, self.sample_rate)

        spec = T.MelSpectrogram(sample_rate=self.sample_rate, n_mels=128, n_fft=400)(waveform)
        spec = T.AmplitudeToDB()(spec)

        return spec, int(label)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    DATASET_PATH = "data/speech_commands_v0.02/dataset.csv"
    NOISE_PATH = "data/speech_commands_v0.02/background_noise.csv"

    dataset = SpeechCommandsDataset(DATASET_PATH, noise_csv=NOISE_PATH)

    for tensor, label in dataset:
        plt.imshow(tensor[0])
        plt.colorbar()
        plt.tight_layout()
        plt.show()
