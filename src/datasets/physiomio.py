import pathlib
from typing import TypedDict

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from src.utils.augmentations import Augmentations
from src.utils.normalization import Normalizer
from src.utils.signal_processing import calculate_window_indices, rms_transform


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

        print(f"Normalizing sEMG signals with {normalizer.__class__.__name__}...")
        self.raw_emgs = normalizer(self.raw_emgs) if normalizer else self.raw_emgs

        self.rms_emgs = None
        if rms_opts:
            print("RMS norming sEMG signals...")
            self.rms_emgs = rms_transform(self.raw_emgs, **rms_opts)

        print("Calculating sEMG window indicies...")
        self.window_indices = calculate_window_indices(
            self.rms_emgs if self.rms_emgs is not None else self.raw_emgs, **window_opts
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
            if self.rms_emgs is not None
            else self.raw_emgs[trial_idx, :, start:end]
        )  # (channels, time_steps)
        gesture = self.gestures[trial_idx]  # (1,)

        emg_window = (
            self.augmentations.time_augment(emg_window)
            if self.augmentations
            else emg_window
        )  # (channels, time_steps)

        return emg_window, gesture


def main():
    import argparse

    from src.utils.registers import NORMALIZERS

    parser = argparse.ArgumentParser(description="Test Phsyio")
    parser.add_argument(
        "--data",
        type=str,
        help="Directory containing the preprocessed patient data files in hdf5 format.",
        required=True,
    )
    parser.add_argument(
        "--patient_ids",
        type=int,
        nargs="+",
        required=True,
        help="Specific patients to load in the dataset",
    )
    parser.add_argument(
        "--norm",
        type=str,
        required=True,
        choices=NORMALIZERS.keys(),
        help="Which normalizer to use on the data",
    )

    args = parser.parse_args()

    normalizer = NORMALIZERS[args.norm]()
    augmentations = Augmentations()

    dataset = PhysioMioDataset(
        pathlib.Path(args.data),
        args.patient_ids,
        normalizer=normalizer,
        augmentations=augmentations,
    )

    print("\n=== Dataset summary ===")
    print(f"Num trials: {len(dataset.raw_emgs)}")
    print(f"Raw EMG shape: {tuple(dataset.raw_emgs.shape)}")
    print(f"Num windows: {len(dataset)}")

    print("\n=== Normalization sanity ===")
    raw = dataset.raw_emgs
    print(f"Per-channel mean (should be ~0 if z-scored): {raw.mean(dim=(0, 2))}")
    print(f"Per-channel std  (should be ~1 if z-scored): {raw.std(dim=(0, 2))}")
    print(
        f"Any NaN: {torch.isnan(raw).any().item()}  Any Inf: {torch.isinf(raw).any().item()}"
    )

    print("\n=== Single-item check ===")
    idx = 1
    emg_window, gesture = dataset[idx]
    print(f"emg_window shape: {tuple(emg_window.shape)}, dtype: {emg_window.dtype}")
    print(f"gesture: {gesture}")
    print(
        f"emg_window mean/std/min/max: "
        f"{emg_window.mean():.4f} / {emg_window.std():.4f} / "
        f"{emg_window.min():.4f} / {emg_window.max():.4f}"
    )

    print("\n=== Augmentation sanity (confirm it actually changes data) ===")
    window_info = dataset.window_indices[idx]
    trial_idx, start, end = (
        window_info["trial_idx"],
        window_info["start"],
        window_info["end"],
    )
    unaugmented = (
        dataset.rms_emgs[trial_idx, :, start:end]
        if dataset.rms_emgs is not None
        else dataset.raw_emgs[trial_idx, :, start:end]
    )
    if dataset.augmentations is not None:
        augmented = dataset.augmentations.time_augment(unaugmented.clone())
        diff = (augmented - unaugmented).abs()
        print(f"Mean abs diff after augmentation: {diff.mean():.6f}")
        print(f"Max abs diff after augmentation:  {diff.max():.6f}")
        if diff.mean() < 1e-6:
            print(
                "WARNING: augmentation produced (near-)identical output — check it's actually applying"
            )
    else:
        print("No augmentations configured, skipping")

    print("\n=== Batch-level check via DataLoader ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_emgs, batch_gestures = next(iter(loader))
    print(f"Batch emg shape: {tuple(batch_emgs.shape)}")
    print(f"Batch gesture shape: {tuple(batch_gestures.shape)}")
    print(
        f"Batch any NaN: {torch.isnan(batch_emgs).any().item()}\nAny Inf: {torch.isinf(batch_emgs).any().item()}"
    )
    print(f"Unique gestures in batch: {torch.unique(batch_gestures).tolist()}")


if __name__ == "__main__":
    main()
