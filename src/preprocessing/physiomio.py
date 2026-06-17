import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pathlib
import shutil

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

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
TIME_PER_WINDOW = 0.5  # seconds
TARGET_LENGTH = FS * TIME_PER_TRIAL  # 8192 samples
CHANNEL_COLS = [f"channel_{i:02d}" for i in range(1, 65)]  # channel_01 to channel_64
WINDOW_SIZE = int(FS * TIME_PER_WINDOW)  # samples
STRIDE = WINDOW_SIZE // 2  # 50% overlap


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
    file_path: pathlib.Path, grouped_labels: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emgs = []
    gestures = []
    fma_scores = []

    patient_data = pd.read_parquet(file_path)
    patient_data["gesture"] = patient_data["movement_type"].map(
        GROUPED_GESTURES_MAP if grouped_labels else UNGROUPED_GESTURES_MAP
    )
    patient_data.fillna({"fma": -1}, inplace=True)
    patient_data["fma"] = patient_data["fma"].map(FMA_MAP)

    value_counts = patient_data["movement_type"].value_counts()
    for movement_type, count in value_counts.items():
        subset = patient_data.loc[
            (patient_data["movement_type"] == movement_type)
            & (patient_data["fma"] != 1)
        ]
        if subset.empty:
            continue

        gestures.append(subset["gesture"].iloc[0])
        fma_scores.append(subset["fma"].iloc[0])
        emg = subset[CHANNEL_COLS].to_numpy(dtype="float32").T

        emg = bandpass_filter(emg)
        emg = notch_filter(emg)

        if emg.shape[1] == 8193:
            emg = emg[:, :TARGET_LENGTH]

        emgs.append(emg)

    return np.array(emgs), np.array(gestures), np.array(fma_scores)


def load_all_patient_data(
    metadata: pd.DataFrame,
    data_dir: pathlib.Path,
    id: int,
    arm_type: str,
    grouped_labels: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arm_type == "both":
        rows = metadata[metadata["patient"] == f"patient{id}"]
    else:
        rows = metadata[
            (metadata["patient"] == f"patient{id}") & (metadata["arm_type"] == arm_type)
        ]

    file_paths = rows["file_path"].tolist()

    patient_emgs = []
    patient_gestures = []
    patient_fmas = []

    for file_path in file_paths:
        patient_emg, patient_label, patient_fma = load_patient_data(
            data_dir / file_path, grouped_labels
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
    grouped_labels: bool,
):
    print(
        f"Loading and preprocessing patient data with grouped_labels={grouped_labels} ..."
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
                grouped_labels,
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


def build_output_dir(base_dir: pathlib.Path, args) -> pathlib.Path:
    parts = []

    parts.append(args.arm_type)

    parts.append("grouped" if args.grouped_labels else "ungrouped")

    folder_name = "_".join(parts)
    return base_dir / folder_name


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
        "--grouped_labels",
        action="store_true",
        help="Whether to use grouped gesture labels (4 classes) instead of ungrouped (16 classes).",
    )

    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    output_dir = build_output_dir(pathlib.Path(args.output_dir), args)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data_dir / "metadata.csv")

    save_patients(
        metadata,
        data_dir,
        output_dir,
        args.arm_type,
        args.grouped_labels,
    )
