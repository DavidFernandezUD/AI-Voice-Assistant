"""
Load dataset for command recognition model.
"""


import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple
import random
import csv


class PadTruncate(nn.Module):
    """
    Pad or truncate a signal to the desired length in milliseconds.

    Args:
        length_ms (int): Desired length.
        sample_rate (int): Sample rate of the signal.
    """

    def __init__(self, length_ms: int, sample_rate: int):
        super(PadTruncate, self).__init__()

        self.length = (sample_rate // 1000) * length_ms

    def forward(self, waveform: Tensor):

        # Truncate signal
        if waveform.shape[1] > self.length:
            waveform = waveform[:, :self.length]

        # Randomly padd beginning and ending of signal (kind of data augmentation?)
        if waveform.shape[1] < self.length:
            pad_begin = random.randint(0, self.length - waveform.shape[1])
            pad_end= self.length - waveform.shape[1] - pad_begin

            waveform = F.pad(waveform, (pad_begin, pad_end), mode="constant", value=0)

        return waveform


class Featurizer(nn.Module):
    """
    Converts a 1d waveform tensor to a mel spectrogram.

    Args:
        orig_sample_rate (int): Original sample rate of the signal
        sample_rate (int): Desired sample rate.
        length_ms (int or None, optional): Desired length in milliseconds.
            If None, the signal won't be resized. (Default: None)
        noise (list or None, optional): List to paths of noise files. (Default: None)
        noise_p (float, optional): Probability that noise will be added to a signal. (Default: 0.5)
        augment (bool, optional): Apply SpecAugment to the signal. (Default: True)
        n_time_masks (int, optional): Number of time masks. (Default: 2)
        n_freq_masks (int, optional): Number of frequency masks. (Default: 2)
        mask_p (float, optional): Maximum proportion of timesteps and frequencies that can be masked (Default: 0.2)
"""

    def __init__(
            self,
            orig_sample_rate: int,
            sample_rate: int,
            length_ms: int = None,
            noise: list = None,
            noise_p: float = 0.5,
            augment: bool = True,
            n_time_masks: int = 2,
            n_freq_masks: int = 2,
            mask_p: float = 0.2
        ):
        super(Featurizer, self).__init__()

        self.noise = noise
        self.noise_p = noise_p

        self.resample = T.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)
        self.mel = T.MelSpectrogram(sample_rate=sample_rate, n_mels=96, n_fft=512)
        self.amp_to_db = T.AmplitudeToDB()
        
        self.augment = None
        if augment:
            self.augment = T.SpecAugment(
                n_time_masks=n_time_masks,
                time_mask_param=10,
                n_freq_masks=n_freq_masks,
                freq_mask_param=10,
                p=mask_p
            )

        self.pad_truncate = None
        if length_ms is not None:
            self.pad_truncate = PadTruncate(length_ms=length_ms, sample_rate=orig_sample_rate)

    def forward(self, waveform: Tensor):

        # Pad or truncate the signal to the desired length
        if self.pad_truncate is not None:
            waveform = self.pad_truncate(waveform)

        # Add background noise if noise available and with 50% chance
        if self.noise is not None and random.random() < self.noise_p:
            noise_rate = random.random() * 0.1
            noise_wave, _ = torchaudio.load(random.choice(self.noise), normalize=True)
            noise_wave = noise_wave[:, :waveform.shape[1]] * noise_rate
            
            waveform  += noise_wave

        waveform = self.resample(waveform)

        spec = self.mel(waveform)
        spec = self.amp_to_db(spec)

        # Apply time and frequency masking
        if self.augment is not None:
            spec = self.augment(spec)

        return spec


class SpeechCommandsDataset(Dataset):

    def __init__(
            self,
            data_csv: str,
            noise_csv: str = None,
            orig_sample_rate: int = 16000,
            sample_rate: int = 16000,
            length_ms: int = None,
            augment: bool = True
        ):

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

        self.featurizer = Featurizer(
            orig_sample_rate=orig_sample_rate,
            sample_rate=sample_rate,
            length_ms=length_ms,
            noise=self.noise,
            noise_p=0.5,
            augment=augment,
            n_time_masks=2,
            n_freq_masks=2,
            mask_p=0.2
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:

        file, label = self.data[index]

        waveform, _ = torchaudio.load(file, normalize=True) # NOTE: normalize=True converts INT audio formats into float32

        spec = self.featurizer(waveform)

        return spec, int(label)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    DATASET_PATH = "data/speech_commands_v0.02/dataset.csv"
    NOISE_PATH = "data/speech_commands_v0.02/background_noise.csv"

    dataset = SpeechCommandsDataset(DATASET_PATH, noise_csv=NOISE_PATH, length_ms=3000)

    for tensor, label in dataset:
        plt.imshow(tensor[0])
        plt.colorbar()
        plt.tight_layout()
        plt.show()
