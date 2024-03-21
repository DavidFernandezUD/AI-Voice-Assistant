import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from audio_utils import pad_truncate, spec_augment, mel_spectrogram
import os
import random



class Featurizer(nn.Module):

    def __init__(self, original_sample_rate, sample_rate: int = 8000, length_ms: int = 1000, n_mels: int = 64, n_fft: int = 400):
        super(Featurizer, self).__init__()

        self.sample_rate = sample_rate
        self.length_ms = length_ms
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.resample = T.Resample(orig_freq=original_sample_rate, new_freq=sample_rate)
        self.noise = self.load_noise("data/speech_commands_v0.02/_background_noise_")

    def load_noise(self, path: str):
        noise = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            audio, sr = torchaudio.load(file_path)
            audio = T.Resample(orig_freq=sr, new_freq=self.sample_rate)(audio)

            noise.append(audio)

        return noise

    def forward(self, x):

        x = self.resample(x)
        x = pad_truncate(x, self.length_ms)

        # Adding random noise
        snr_dbs = torch.tensor([10, 10])
        length = x[0].shape[1]
        noise_sample = random.choice(self.noise)[:, :length]
        signal = T.AddNoise()(x[0], noise_sample, snr=snr_dbs)
        x = (signal, x[1])

        x = mel_spectrogram(x, channels=self.n_mels, n_fft=self.n_fft)
        x = spec_augment(x, n_freq_masks=2, n_time_masks=1)

        return x
    

class SpeechCommandsDataset(Dataset):

    def __init__(self, path: str, sample_rate: int = 16000):

        self.valid_commands = [
            "Yes", "No", "Up", "Down", "Left",
            "Right", "On", "Off", "Stop", "Go",
            "Zero", "One", "Two", "Three", "Four",
            "Five", "Six", "Seven", "Eight", "Nine"
        ]

        self.invalid_commands = [
            "Bed", "Bird", "Cat", "Dog", "Happy",
            "House", "Marvin", "Sheila", "Tree", "Wow"
        ]

        self.sample_rate = sample_rate
        self.featurizer = Featurizer(16000, sample_rate=sample_rate, length_ms=2000)

        self.X, self.y = self.load_data(path)

    def load_data(self, path: str):
        X = []
        y = []
        for dir in os.listdir(path):
            dir_path = os.path.join(path, dir)
            if os.path.isdir(dir_path) and dir != "_background_noise_":
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)

                    audio = torchaudio.load(file_path, normalize=True)
                    spec = self.featurizer(audio)

                    X.append(spec)

                    if dir in self.valid_commands:
                        y.append(self.valid_commands.index(dir))
                    else:
                        # all invalid commands go into one class
                        y.append(20)

        return X, y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":

    dataset = SpeechCommandsDataset("data/speech_commands_v0.02")

    import matplotlib.pyplot as plt

    for x in dataset.X:
        plt.imshow(x[0])
        plt.show()

    print(dataset.X[0].shape)
