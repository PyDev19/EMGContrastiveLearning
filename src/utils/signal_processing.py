from typing import TypedDict

import torch


class WindowIndex(TypedDict):
    trial_idx: int
    start: int
    end: int

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
