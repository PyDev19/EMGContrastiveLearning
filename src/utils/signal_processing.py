from typing import TypedDict

import numpy as np
import torch
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


class WindowIndex(TypedDict):
    trial_idx: int
    start: int
    end: int


def bandpass_filter(
    emg, order=4, low_cutoff=20, high_cutoff=500, fs=2048
) -> np.ndarray:
    sos = butter(order, [low_cutoff, high_cutoff], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, emg, axis=1)


def notch_filter(emg, quality_factor=30, notch_freq=50, fs=2048) -> np.ndarray:
    b, a = iirnotch(notch_freq, quality_factor, fs=fs)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, emg, axis=1)


def calculate_window_indices(
    signal: torch.Tensor, size: int, stride: int
) -> list[WindowIndex]:
    window_index = []

    for trial_idx in range(len(signal)):
        trial_length = signal[trial_idx].shape[1]

        for start_init in range(0, trial_length - size + 1, stride):
            window_index.append(
                {
                    "trial_idx": trial_idx,
                    "start": start_init,
                    "end": start_init + size,
                }
            )

    return window_index


def rms_transform(signal: torch.Tensor, size: int, stride: int):
    windows = signal.unfold(-1, size, stride)
    return torch.sqrt(torch.mean(windows**2, dim=-1))
