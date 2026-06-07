import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pathlib
import shutil

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.signal import resample, butter, iirnotch, sosfiltfilt, tf2sos

GROUPED_GESTURES_MAP = {
    "Rest": 0,
    "MassFlexion": 1,
    "HookGrasp": 1,
    "DiameterGrasp": 1,
    "SphereGrasp": 1,
    "PinchGrasp": 2,
    "PinchGraspMiddle": 2,
    "PinchGraspRing": 2,
    "PinchGraspPinkie": 2,
    "ThumbAdduction": 3,
    "MassExtension": 4,
    "MassAdduction": 5,
    "WristVolarFlexion": 6,
    "WristDorsiFlexion": 7,
    "ForearmPronation": 8,
    "ForearmSupination": 9,
}

UNGROUPED_GESTURES_MAP = {
    "Rest": 0,
    "MassFlexion": 1,
    "HookGrasp": 2,
    "DiameterGrasp": 3,
    "SphereGrasp": 4,
    "PinchGrasp": 5,
    "PinchGraspMiddle": 6,
    "PinchGraspRing": 7,
    "PinchGraspPinkie": 8,
    "ThumbAdduction": 9,
    "MassExtension": 10,
    "MassAdduction": 11,
    "WristVolarFlexion": 12,
    "WristDorsiFlexion": 13,
    "ForearmPronation": 14,
    "ForearmSupination": 15,
}

FMA_MAP = {
    -1: 0,
    0: 1,
    1: 2,
    2: 3,
}

FS = 2048  # hz
LOW_CUTOFF = 20  # hz
HIGH_CUTOFF = 500  # hz
NOTCH_FREQ = 50  # hz
RMS_WINDOW_MS = 100  # ms
WORKERS = os.cpu_count() - 1  # cpu cores
TIME_PER_TRIAL = 4  # seconds
TARGET_LENGTH = FS * TIME_PER_TRIAL  # 8192 samples


def bandpass_filter(emg, order=4) -> np.ndarray:
    sos = butter(order, [LOW_CUTOFF, HIGH_CUTOFF], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, emg, axis=1)


def notch_filter(emg, quality_factor=30) -> np.ndarray:
    b, a = iirnotch(NOTCH_FREQ, quality_factor, fs=FS)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, emg, axis=1)


def rms_envelope(
    emg: np.ndarray, fs: int = FS, window_ms: float = RMS_WINDOW_MS
) -> tuple[np.ndarray, int, int]:
    window_samples = int(fs * window_ms / 1000)  # 2048 * 0.1 = 205 samples
    n_channels, n_samples = emg.shape

    n_windows = n_samples // window_samples
    emg_trimmed = emg[:, : n_windows * window_samples]  # (C, n_windows * W)

    emg_blocks = emg_trimmed.reshape(n_channels, n_windows, window_samples)
    rms_emg = np.sqrt(np.mean(emg_blocks**2, axis=-1))  # (C, n_windows)

    return rms_emg, n_windows, window_samples


def load_patient_data(
    file_path: pathlib.Path, rms: bool, grouped_labels: bool, rms_window_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_data = pd.read_parquet(file_path)
    patient_data["gesture"] = patient_data["movement_type"].map(
        GROUPED_GESTURES_MAP if grouped_labels else UNGROUPED_GESTURES_MAP
    )
    patient_data.fillna({"fma": -1}, inplace=True)
    patient_data["fma"] = patient_data["fma"].map(FMA_MAP)

    times = patient_data["time"].to_numpy()
    time_diffs = np.abs(np.diff(times, prepend=times[0]))
    boundary_mask = np.round(time_diffs) >= 4
    trial_ids = np.cumsum(boundary_mask)

    unique, counts = np.unique(trial_ids, return_counts=True)

    channel_cols = [f"channel_{i:02d}" for i in range(1, 65)]
    emg_array = patient_data[channel_cols].to_numpy(dtype=np.float32)
    gestures_array = patient_data["gesture"].to_numpy()
    fmas_array = patient_data["fma"].to_numpy()

    all_trial_emgs = []
    all_trial_gestures = []
    all_trial_fmas = []

    unique_trials = np.unique(trial_ids)

    for tid in unique_trials:
        mask = trial_ids == tid
        trial_emg = emg_array[mask].T
        trial_gestures = gestures_array[mask]
        trial_fmas = fmas_array[mask]

        trial_emg = bandpass_filter(trial_emg)
        trial_emg = notch_filter(trial_emg)

        # majority label for gesture and fma in the trial
        trial_gesture = stats.mode(trial_gestures, keepdims=True).mode[0]
        trial_fma = stats.mode(trial_fmas, keepdims=True).mode[0]

        original_length = trial_emg.shape[1]
        if original_length != TARGET_LENGTH:
            trial_emg = resample(trial_emg, TARGET_LENGTH, axis=1)

        if rms:
            trial_emg, n_windows, window_samples = rms_envelope(
                trial_emg, window_ms=rms_window_size
            )

        all_trial_emgs.append(trial_emg)
        all_trial_gestures.append(trial_gesture)
        all_trial_fmas.append(trial_fma)

    return (
        np.array(
            all_trial_emgs
        ),  # shape: (num_trials, 64, TARGET_LENGTH / RMS_WINDOW_MS)
        np.array(all_trial_gestures),  # shape: (num_trials, 1)
        np.array(all_trial_fmas),  # shape: (num_trials, 1)
    )


def load_all_patient_data(
    metadata: pd.DataFrame,
    data_dir: pathlib.Path,
    id: int,
    arm_type: str,
    rms: bool,
    grouped_labels: bool,
    rms_window_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arm_type == "both":
        rows = metadata[metadata["patient"] == f"patient{id}"]
    else:
        rows = metadata[
            (metadata["patient"] == f"patient{id}")
            & (metadata["arm_type"] == arm_type)
        ]

    file_paths = rows["file_path"].tolist()

    patient_emgs = []
    patient_gestures = []
    patient_fmas = []

    for file_path in file_paths:
        patient_emg, patient_label, patient_fma = load_patient_data(
            data_dir / file_path, rms, grouped_labels, rms_window_size
        )

        patient_emgs.append(patient_emg)
        patient_gestures.append(patient_label)
        patient_fmas.append(patient_fma)

    return (
        np.concatenate(patient_emgs, axis=0),  # shape: (num_trials, 64, TARGET_LENGTH)
        np.concatenate(patient_gestures, axis=0),  # shape: (num_trials, 1)
        np.concatenate(patient_fmas, axis=0),  # shape: (num_trials, 1)
    )


def save_patients(
    metadata: pd.DataFrame,
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    arm_type: str,
    rms: bool,
    grouped_labels: bool,
    rms_window_size: float,
):
    print(
        f"Loading and preprocessing patient data with RMS={rms}, grouped_labels={grouped_labels}, rms_window_size={rms_window_size}ms ..."
    )

    # Parallel load with a tqdm progress bar
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_idx = {
            executor.submit(
                load_all_patient_data,
                metadata,
                data_dir,
                i,
                arm_type,
                rms,
                grouped_labels,
                rms_window_size,
            ): i
            for i in range(1, 49)
        }

        with tqdm(total=48, desc="Loading patients", unit="patient") as pbar:
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                emgs, gestures, fmas = future.result()
                with h5py.File(output_dir / f"patient_{i}.h5", "w") as f:
                    f.create_dataset(
                        "emgs",
                        data=emgs,
                        compression="gzip",
                        compression_opts=4,
                    )
                    f.create_dataset(
                        "gestures",
                        data=gestures,
                        compression="gzip",
                        compression_opts=4,
                    )
                    f.create_dataset(
                        "fmas", data=fmas, compression="gzip", compression_opts=4
                    )
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and save patient EMG data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/patient_data",
        help="Directory containing the raw patient data files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed",
        help="Directory to save the preprocessed patient data.",
    )
    parser.add_argument(
        "--arm_type",
        type=str,
        default="impaired_arm",
        choices=["impaired_arm", "healthy_arm", "both"],
        help="Type of arm for which to process data (default: impaired_arm).",
    )
    parser.add_argument(
        "--rms",
        action="store_true",
        help="Whether to apply RMS envelope to the EMG data.",
    )
    parser.add_argument(
        "--grouped_labels",
        action="store_true",
        help="Whether to use grouped gesture labels (4 classes) instead of ungrouped (16 classes).",
    )
    parser.add_argument(
        "--rms_window_size",
        type=float,
        default=RMS_WINDOW_MS,
        help="RMS window size in milliseconds.",
    )

    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not args.rms:
        output_dir = pathlib.Path(f"{args.output_dir}/raw/")
    else:
        output_dir = pathlib.Path(f"{args.output_dir}/rms_{args.rms_window_size}ms/")

    if args.grouped_labels:
        output_dir = output_dir.parent / f"{output_dir.stem}_grouped_labels"
    else:
        output_dir = output_dir.parent / f"{output_dir.stem}_ungrouped_labels"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data_dir / "metadata.csv")

    save_patients(
        metadata,
        data_dir,
        output_dir,
        args.arm_type,
        args.rms,
        args.grouped_labels,
        args.rms_window_size,
    )
