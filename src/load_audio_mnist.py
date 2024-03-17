import pandas as pd
import torchaudio
import os

import numpy as np
import matplotlib.pyplot as plt


data_path = "data/AudioMNIST/"
digits = 10
sample_rate = 8000


def load_audioMNIST(path, digits, sample_rate):

    data = {"audio": [], "label": []}

    for dir in os.listdir(path):
        speaker_dir = os.path.join(path, dir)
        for file in os.listdir(speaker_dir):
            file_path = os.path.join(speaker_dir, file)

            waveform, _ = torchaudio.load(file_path, normalize=True)
            digit = file[0]

            data["audio"].append(waveform)
            data["label"].append(int(digit))

    return data


if __name__ == "__main__":

    data = load_audioMNIST(data_path, digits, sample_rate)

    samples = data["audio"][:100]

    for sample in samples:

        n_frames = sample.shape[1]
        time_axis = np.arange(0, n_frames) / sample_rate
        
        plt.specgram(sample[0], Fs=sample_rate)
        plt.show()
