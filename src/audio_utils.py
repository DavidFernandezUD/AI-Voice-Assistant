import torchaudio
from torchaudio import transforms
import torch
from torch import Tensor
from typing import Tuple
import random


def resample(audio: Tuple[Tensor, int], sample_rate: int) -> Tuple[Tensor, int]:
    """
    Resamples an audio into the specified sample rate.

    Parameters
    ----------
    audio : Tuple[Tensor, int]
        Audio tensor with shape (channels, smples) and sample rate.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    Tuple[Tensor, int]
        Resampled tensor and new sample rate.
    """

    signal, freq = audio

    signal = torchaudio.transforms.Resample(
        orig_freq=freq,
        new_freq=sample_rate
    )(signal)

    return (signal, sample_rate)


def pad_truncate(audio: Tuple[Tensor, int], length_ms: int) -> Tuple[Tensor, int]:
    """
    Truncate or pad an audio tensor to the specified length in milliseconds.
    In case of padding, the signal is paded both at begining and end randomly.

    Parameters
    ----------
    audio : Tuple[Tensor, int]
        Audio tensor with shape (channels, smples) and sample rate.
    length : int
        Desired length of the audio tensor ini milliseconds.
            
    Returns
    -------
    Tuple[Tensor, int]
        Trunkated or padded tensor of desired length and sample rate.
    """

    signal, freq = audio

    max_len = (freq // 1000) * length_ms

    # Truncating signal
    if signal.shape[1] > max_len:
        signal = signal[:, :max_len]

    # Randomly padding beginning and ending of signal (kind of data augmentation?)
    if signal.shape[1] < max_len:
        pad_begin_len = random.randint(0, max_len - signal.shape[1])
        pad_end_len = max_len - signal.shape[1] - pad_begin_len

        pad_begin = torch.zeros((signal.shape[0], pad_begin_len))
        pad_end = torch.zeros((signal.shape[0], pad_end_len))

        signal = torch.cat((pad_begin, signal, pad_end), dim=1)

    return (signal, freq)


def mel_spectrogram(audio: Tuple[Tensor, int], channels: int=64, n_fft: int=512):
    """
    Converts audio tensor into a mel spectogram and adjusts amplitude to db scale.

    Parameters
    ----------
    audio : Tuple[Tensor, int]
        Audio tensor with shape (channels, smples) and sample rate.
    channels : int
        MEL channels or bins of the spectrogram.
    n_fft : int
        Resolution of the spectrogram in the time axis.
    
    Returns
    -------
    Tensor
        Spectrogram of the original audio signal adjusted to mel and db scales.
    """

    signal, freq = audio

    # Convert audio signal to mel spectrogram
    spectrogram = transforms.MelSpectrogram(sample_rate=freq, n_mels=channels, n_fft=n_fft)(signal)

    # Convert amplitude to db scale
    spectrogram = transforms.AmplitudeToDB()(spectrogram)

    return spectrogram


def MFCC_spectrogram(audio: Tuple[Tensor, int], channels: int=64):
    """
    Converts audio tensor into a Mel Frequency Cepstral Coefficients spectogram
    and adjusts amplitude to db scale.

    Parameters
    ----------
    audio : Tuple[Tensor, int]
        Audio tensor with shape (channels, smples) and sample rate.
    channels : int
        MFCC channels or bins of the spectrogram.
    
    Returns
    -------
    Tensor
        MFCC spectrogram adjusted to db scale.
    """

    signal, freq = audio

    # Convert audio signal to mel spectrogram
    spectrogram = transforms.MFCC(sample_rate=freq, n_mfcc=channels)(signal)

    # Convert amplitude to db scale
    spectrogram = transforms.AmplitudeToDB()(spectrogram)

    return spectrogram


def spec_augment(spectrogram: Tensor, max_mask: float=0.1, n_freq_masks: int=1, n_time_masks: int=1):
    """
    Masks bands in both the time and frequency dimension of a spectrogram.

    Parameters
    ----------
    spectrogram : Tensor
        Spectogram tensor with shape (frequency, time).
    max_mask : float
        Maximum possible length of each mask.
    n_freq_masks : int
        Number of discrete masks in the frequency dimension.
    n_time_masks : int
        Number of discrete masks in the time dimension.

    Returns
    -------
    Tensor
        Masked spectrogram.
    """    

    _, n_channels, n_samples = spectrogram.shape
    mask_value = spectrogram.mean()

    # Masking frequency dimension
    for _ in range(n_freq_masks):
        spectrogram = transforms.FrequencyMasking(max_mask * n_channels)(spectrogram, mask_value)

    # Masking time dimension
    for _ in range(n_time_masks):
        spectrogram = transforms.TimeMasking(max_mask * n_samples)(spectrogram, mask_value)

    return spectrogram


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import os

    PATH = "data/AudioMNIST/01/"
    SAMPLE_RATE = 8000
    LENGTH = 1000

    for file in os.listdir(PATH):
        file_path = os.path.join(PATH, file)

        audio = torchaudio.load(file_path)

        audio = resample(audio, SAMPLE_RATE)
        audio = pad_truncate(audio, LENGTH)

        spec = mel_spectrogram(audio, 64, 400)
        # spec = MFCC_spectrogram(audio, 64)

        # spec = spec_augment(spec, max_mask=0.1, n_freq_masks=2, n_time_masks=1)

        print(f"shape: {spec.shape}")

        plt.imshow(spec[0])
        plt.show()
