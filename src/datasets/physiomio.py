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
        jitter_sigma: float = 0.8,
        scale_sigma: float = 1.1,
        mask_prob: float = 0.5,
        freq_perturb_ratio: float = 0.25,
        freq_alpha: float = 0.1,
    ):
        self.emgs = []
        self.gestures = []

        for patient_id in patient_ids:
            with h5py.File(data_dir / f"patient_{patient_id}.h5", "r") as f:
                emgs = f["emgs"][:]
                gestures = f["gestures"][:]

                self.emgs.append(emgs)
                self.gestures.append(gestures)

        self.emgs = np.concatenate(self.emgs, axis=0)  # (trials, channels, time steps)
        self.gestures = np.concatenate(self.gestures, axis=0)  # (trials, time steps)

        if mean is None:
            self.mean = self.emgs.mean(axis=(0, 2), keepdims=True)
        else:
            self.mean = mean

        if std is None:
            self.std = self.emgs.std(axis=(0, 2), keepdims=True) + 1e-8
        else:
            self.std = std

        self.emgs = (self.emgs - self.mean) / self.std

        self.augmentations = Augmentations(
            jitter_sigma=jitter_sigma,
            scale_sigma=scale_sigma,
            mask_prob=mask_prob,
            freq_perturb_ratio=freq_perturb_ratio,
            freq_alpha=freq_alpha,
        )

    def __len__(self):
        return len(self.emgs)

    def __getitem__(self, index):
        emg_window = self.emgs[index]  # (channels, window_size)
        gesture = torch.tensor(self.gestures[index], dtype=torch.long)  # (1,)

        emg_window = torch.from_numpy(emg_window).float()
        fft_window = torch.abs(fft(emg_window, dim=-1))  # (channels, window_size)

        time_augmented_window = self.augmentations.time_augment(emg_window.clone())

        frequency_augmented_fft = self.augmentations.frequency_augment(
            fft_window.clone()
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

    patient_ids = list(range(1, 39))
    dataset = PhysioMioEMGDataset(
        data_dir, patient_ids, window_size=512, stride=256, mask_prob=0.2
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
