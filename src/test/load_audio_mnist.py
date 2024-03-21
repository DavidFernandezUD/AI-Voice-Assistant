import torchaudio
import torch
import os
import matplotlib.pyplot as plt
from audio_utils import resample, pad_truncate, mel_spectrogram, spec_augment


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

    specs = []
    digits = []
    for dir in os.listdir(path):
        speaker_dir = os.path.join(path, dir)
        for file in os.listdir(speaker_dir):
            file_path = os.path.join(speaker_dir, file)

            audio = torchaudio.load(file_path, normalize=True)
            audio = resample(audio, sample_rate)
            audio = pad_truncate(audio, length_ms)

            spec = mel_spectrogram(audio, 64, 400)
            spec = spec_augment(spec, max_mask=0.1, n_freq_masks=2, n_time_masks=0)

            specs.append(spec)
            digits.append(torch.tensor(int(file[0])))

    return torch.stack(specs), torch.stack(digits)


if __name__ == "__main__":

    DATA_PATH = "data/AudioMNIST/"
    SAMPLE_RATE = 8000
    LENGTH = 1000

    data = load_audioMNIST(DATA_PATH, SAMPLE_RATE, LENGTH)

    audios = data["audio"][:100]

    for audio in audios:
        
        sample, freq = audio

        n_frames = sample.shape[1]
        time_axis = torch.arange(0, n_frames) / SAMPLE_RATE

        print(sample)
        
        plt.specgram(sample[0], Fs=SAMPLE_RATE)
        plt.show()
