import torchaudio
import torch
from torch import Tensor
import os
import matplotlib.pyplot as plt
from audio_utils import resample, pad_truncate


def load_audioMNIST(path: str, sample_rate: int, length_ms: int):
    """
    Load audioMNIST dataset into a dictionary with keys "audio" and "label".

    Parameters
    ----------
    path : str
        Path of the audioMNIST directory.
    sample_rate : int
        Sample rate the audio files will be transformed to.
    length_ms : int
        Length in milliseconds the audios will be padded to.

    Returns
    -------
    Tuple[List[Tuple[Tensor, int]], List[int]]
        Tuple with audios (signal and sample rate) and labels.
    """

    data = ([], [])
    for dir in os.listdir(path):
        speaker_dir = os.path.join(path, dir)
        for file in os.listdir(speaker_dir):
            file_path = os.path.join(speaker_dir, file)

            audio = torchaudio.load(file_path, normalize=True)
            audio = resample(audio, sample_rate)
            audio = pad_truncate(audio, length_ms)

            digit = file[0]

            data[0].append(audio)
            data[1].append(int(digit))

    return data


if __name__ == "__main__":

    data_path = "data/AudioMNIST/"
    digits = 10
    sample_rate = 8000
    length = 1000

    data = load_audioMNIST(data_path, sample_rate, length)

    audios = data["audio"][:100]

    for audio in audios:
        
        sample, freq = audio

        n_frames = sample.shape[1]
        time_axis = torch.arange(0, n_frames) / sample_rate

        print(sample)
        
        plt.specgram(sample[0], Fs=sample_rate)
        plt.show()
