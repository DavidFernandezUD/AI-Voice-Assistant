import torchaudio
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


def mel_spectogram(audio: Tuple[Tensor, int]):
    # TODO: Implement audio to MEL spectogram
    pass
