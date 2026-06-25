import argparse
import pathlib
from time import time

import h5py
import numpy as np
import torch
from torch.fft import fft
from torch.utils.data import DataLoader, Dataset

from src.augmentations import Augmentations


class PhysioMioEMGDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        patient_ids: list[int],
        mean: np.ndarray = None,
        std: np.ndarray = None,
        augmentations: Augmentations = None,
        rms_window_samples: int = None,
        rms_window_stride: int = None,
        window_size: int = 512,
        stride: int = 256,
    ):
        self.emgs = []
        self.gestures = []
        self.rms_window_samples = rms_window_samples
        self.rms_window_stride = rms_window_stride
        self.window_size = window_size
        self.stride = stride

        for patient_id in patient_ids:
            with h5py.File(data_dir / f"patient_{patient_id}.h5", "r") as f:
                emgs = f["emgs"][:]
                gestures = f["gestures"][:]

                self.emgs.append(emgs)
                self.gestures.append(gestures)

        self.emgs = np.concatenate(self.emgs, axis=0)  # (trials, channels, time steps)
        self.gestures = np.concatenate(self.gestures, axis=0)  # (trials, 1)

        self.mean = mean
        self.std = std

        if mean is not None and std is not None:
            self.mean = mean.reshape(1, -1, 1)
            self.std = std.reshape(1, -1, 1)
            self.emgs = (self.emgs - self.mean) / self.std

        self.augmentations = augmentations

        self.window_indices = self._calculate_window_indices()

    def __len__(self):
        if self.window_indices is not None:
            return len(self.window_indices)
        return len(self.emgs)

    def _calculate_window_indices(self) -> list[dict[str, int]] | None:
        window_index = []
        for trial_idx in range(len(self.emgs)):
            trial_length = self.emgs[trial_idx].shape[1]
            for start_init in range(
                0, trial_length - self.window_size + 1, self.stride
            ):
                window_index.append(
                    {
                        "trial_idx": trial_idx,
                        "start": start_init,
                        "end": start_init + self.window_size,
                    }
                )

        return window_index if window_index else None

    def _rms_window(self, window: torch.Tensor) -> torch.Tensor:
        if self.rms_window_samples is None:
            return window

        blocks = window.unfold(-1, self.rms_window_samples, self.rms_window_stride)
        return torch.sqrt(torch.mean(blocks**2, dim=-1))  # (channels, n_windows)

    def __getitem__(self, index):
        if self.window_indices is not None:
            window_info = self.window_indices[index]
            trial_idx = window_info["trial_idx"]
            start = window_info["start"]
            end = window_info["end"]

            emg_window = self.emgs[trial_idx, :, start:end]  # (channels, window_size)
            gesture = torch.tensor(self.gestures[trial_idx], dtype=torch.long)  # (1,)
        else:
            emg_window = self.emgs[index]  # (channels, time steps)
            gesture = torch.tensor(self.gestures[index], dtype=torch.long)  # (1,)

        emg_window = torch.from_numpy(emg_window).float()
        fft_window = torch.abs(fft(emg_window, dim=-1))  # (channels, window_size)
        emg_window = self._rms_window(emg_window)  # (channels, rms_windows)

        if self.augmentations is None:
            return emg_window, fft_window, gesture

        time_augmented_window, frequency_augmented_fft = self.augmentations(
            emg_window.clone(), fft_window.clone()
        )

        return (
            emg_window,
            fft_window,
            time_augmented_window,
            frequency_augmented_fft,
            gesture,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PhysioMioEMGDataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/impaired_preprocessed",
        help="Path to preprocessed data directory",
    )

    args = parser.parse_args()
    data_dir = pathlib.Path(args.data_dir)

    patient_ids = list(range(1, 6))

    augmentations = Augmentations()

    dataset = PhysioMioEMGDataset(
        data_dir,
        patient_ids,
        window_size=32,
        stride=16,
        augmentations=augmentations,
        rms_window_samples=10,
        rms_window_stride=5
    )

    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=64, pin_memory=True)
    print(f"Number of batches: {len(dataloader)}")

    start = time()
    for batch in dataloader:
        (
            emg_window,
            fft_window,
            time_augmented_window,
            frequency_augmented_fft,
            gesture,
        ) = batch
        print(f"EMG window shape: {emg_window.shape}")
        print(f"Time-augmented window shape: {time_augmented_window.shape}")
        print(f"EMG FFT shape: {fft_window.shape}")
        print(f"Frequency-augmented FFT shape: {frequency_augmented_fft.shape}")
        print(f"Gesture shape: {gesture.shape}")
        print(f"Batch processing time: {time() - start:.4f} seconds")
        break
