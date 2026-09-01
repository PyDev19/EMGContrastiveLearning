import pathlib
from typing import TypedDict

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.augmentations import Augmentations
from utils.normalization import Normalizer
from utils.signal_processing import calculate_window_indices, rms_transform


class WindowOpts(TypedDict):
    size: int
    stride: int


class PhysioMioDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        patient_ids: list[int],
        window_opts: WindowOpts | None = None,
        rms_opts: WindowOpts | None = None,
        normalizer: Normalizer | None = None,
        augmentations: Augmentations | None = None,
    ):
        if window_opts is None:
            window_opts = {"size": 512, "stride": 256}

        self.augmentations = augmentations
        self.emgs = []
        self.gestures = []

        print("Loading patient data..")
        for patient_id in patient_ids:
            with h5py.File(data_dir / f"patient_{patient_id}.h5", "r") as f:
                emgs = f["emgs"][:]  # pyright: ignore[reportIndexIssue]
                gestures = f["gestures"][:]  # pyright: ignore[reportIndexIssue]

                self.emgs.append(emgs)
                self.gestures.append(gestures)

        print("Concatenating dataset and converting to tensor...")
        self.raw_emgs = torch.from_numpy(
            np.concatenate(self.emgs, axis=0)
        )  # (trials, channels, time steps)
        self.gestures = torch.from_numpy(
            np.concatenate(self.gestures, axis=0)
        ).long()  # (trials, 1)

        print(f"Normalizing sEMG signals with {normalizer.__class__}...")
        self.raw_emgs = normalizer(self.raw_emgs) if normalizer else self.raw_emgs

        self.rms_emgs = None
        if rms_opts:
            print("RMS norming sEMG signals...")
            self.rms_emgs = rms_transform(self.raw_emgs, **rms_opts)

        print("Calculating sEMG window indicies...")
        self.window_indices = calculate_window_indices(
            self.rms_emgs if self.rms_emgs else self.raw_emgs, **window_opts
        )

    def __len__(self):
        if self.window_indices is not None:
            return len(self.window_indices)

        return len(self.emgs)

    def __getitem__(self, index):
        window_info = self.window_indices[index]
        trial_idx = window_info["trial_idx"]
        start = window_info["start"]
        end = window_info["end"]

        emg_window = (
            self.rms_emgs[trial_idx, :, start:end]
            if self.rms_emgs
            else self.raw_emgs[trial_idx, :, start:end]
        )  # (channels, time_steps)
        gesture = self.gestures[trial_idx]  # (1,)

        emg_window = (
            self.augmentations.time_augment(emg_window)
            if self.augmentations
            else emg_window
        )  # (channels, time_steps)

        return emg_window, gesture


if __name__ == "__main__":
    import argparse

    from src.utils.registers import NORMALIZERS

    parser = argparse.ArgumentParser(description="Test Phsyio")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the preprocessed patient data files in hdf5 format.",
        required=True,
    )
    parser.add_argument(
        "--patient_ids",
        type=int,
        nargs="+",
        required=True,
        description="Specific patients to load in the dataset",
    )
    parser.add_argument(
        "--norm",
        type=str,
        required=True,
        description="Which normalizer to use on the data",
    )

    args = parser.parse_args()

    normalizer = NORMALIZERS[args.norm]()
    augmentations = Augmentations()

    dataset = PhysioMioDataset(
        pathlib.Path(args.data_dir),
        args.patient_ids,
        normalizer=normalizer,
        augmentations=augmentations,
    )

    print(*dataset[1])
