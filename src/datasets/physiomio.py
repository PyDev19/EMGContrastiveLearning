import pathlib
import random
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset

from src.utils.augmentations import Augmentations
from src.utils.normalization import Normalizer
from src.utils.registers import DATASETS, SAMPLERS, register
from src.utils.signal_processing import calculate_window_indices, rms_transform
from src.utils.types import NormalizerScopeName, WindowIndex, WindowOpts


@register(SAMPLERS, "physiomio")
class PhysioMioTrialBatchSampler(Sampler[list[int]]):
    def __init__(self, window_indices: list[WindowIndex], shuffle: bool = True, seed: int = 42):
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        trial_to_windows = defaultdict(list)
        for i, window_info in enumerate(window_indices):
            trial_to_windows[int(window_info["trial_idx"])].append(i)

        self.trials = sorted(trial_to_windows)
        self.trial_windows = [trial_to_windows[t] for t in self.trials]

        self.windows_per_trial = min(len(windows) for windows in self.trial_windows)

    def __len__(self):
        return self.windows_per_trial

    def __iter__(self):
        orders = []
        for windows in self.trial_windows:
            if self.shuffle:
                perm = windows[:]
                self.rng.shuffle(perm)
                orders.append(perm)
            else:
                orders.append(windows)

        for i in range(self.windows_per_trial):
            yield [order[i] for order in orders]


@register(DATASETS, "physiomio")
class PhysioMioDataset(Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        patient_ids: list[int],
        window_opts: WindowOpts | None = None,
        rms_opts: WindowOpts | None = None,
        normalizer: Normalizer | None = None,
        normalizer_scope: NormalizerScopeName = "global",
        augmentations: Augmentations | None = None,
    ):
        self.augmentations = augmentations
        self.rms_opts = rms_opts

        if window_opts is None:
            window_opts = {"size": 512, "stride": 256}

        emgs = []
        gestures = []
        subjects = []

        print("Loading patient data...")
        for patient_id in patient_ids:
            with h5py.File(data_dir / f"patient_{patient_id}.h5", "r") as f:
                emg = torch.from_numpy(f["emgs"][:])  # pyright: ignore[reportIndexIssue]
                if normalizer and normalizer_scope == "per-patient":
                    print(
                        f"Normalizing patient {patient_id} sEMG signals with {normalizer.__class__.__name__}..."
                    )
                    emg = normalizer.fit_transform(emg)

                emgs.append(emg)
                gestures.append(f["gestures"][:])  # pyright: ignore[reportIndexIssue]
                subjects.append(np.full((f["gestures"].shape[0],), patient_id))  # type: ignore

        print("Concatenating dataset and converting to tensor...")
        self.raw_emgs = torch.concatenate(emgs, dim=0)  # (trials, channels, time steps)
        self.gestures = torch.from_numpy(
            np.concatenate(gestures, axis=0)
        ).long()  # (trials, 1)
        self.subjects = torch.from_numpy(
            np.concatenate(subjects, axis=0)
        ).long()  # (trials, 1)

        self.subjects_to_idx = {
            p: i
            for i, p in enumerate(torch.unique(self.subjects, sorted=True).tolist())
        }

        if normalizer and normalizer_scope == "global":
            print(
                f"Normalizing global sEMG signals with {normalizer.__class__.__name__}..."
            )
            if normalizer._fitted:
                self.raw_emgs = normalizer.transform_(self.raw_emgs)
            else:
                normalizer.fit(self.raw_emgs)
                self.raw_emgs = normalizer.transform_(self.raw_emgs)

        print("Calculating sEMG window indicies...")
        self.window_indices = calculate_window_indices(self.raw_emgs, **window_opts)

    def __len__(self):
        if self.window_indices is not None:
            return len(self.window_indices)

        return len(self.raw_emgs)

    def __getitem__(self, index):
        window_info = self.window_indices[index]
        trial_idx = window_info["trial_idx"]
        start = window_info["start"]
        end = window_info["end"]

        emg_window = self.raw_emgs[trial_idx, :, start:end]
        emg_window = emg_window.float()  # (channels, time_steps)
        gesture = self.gestures[trial_idx]
        gesture = gesture.long()  # (1,)

        subject = self.subjects[trial_idx].item()
        subject = self.subjects_to_idx[subject]
        subject = torch.tensor(subject, dtype=torch.long)  # (1,)

        emg_window_aug = (
            self.augmentations(emg_window) if self.augmentations else 0
        )  # (channels, time_steps)

        emg_window = (
            rms_transform(emg_window, **self.rms_opts) if self.rms_opts else emg_window
        )

        return emg_window, emg_window_aug, gesture, subject
