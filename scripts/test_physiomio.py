import pathlib

import torch

from src.datasets import PhysioMioDataset
from src.datasets.physiomio import PhysioMioTrialBatchSampler
from src.utils.augmentations import Augmentations
from src.utils.registers import NORMALIZERS


def main():
    import argparse
    import sys

    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Test PhysioMio dataset and sampler")
    parser.add_argument(
        "--data", type=str, required=True, help="Directory with patient_*.h5 files."
    )
    parser.add_argument("--patient_ids", type=int, nargs="+", required=True)
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        choices=list(NORMALIZERS.keys()),
        help="Optional normalizer (omit to test without one).",
    )
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--epochs", type=int, default=2, help="Sampler epochs to iterate."
    )
    args = parser.parse_args()

    failures: list[str] = []

    def check(name: str, ok, detail: str = "") -> None:
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name}"
            + (f"  ({detail})" if detail else "")
        )
        if not ok:
            failures.append(name)

    window_opts = {"size": args.window_size, "stride": args.stride}
    normalizer = NORMALIZERS[args.norm]() if args.norm else None

    print("\n=== Building dataset ===")
    dataset = PhysioMioDataset(
        pathlib.Path(args.data),
        args.patient_ids,
        window_opts=window_opts,  # pyright: ignore[reportArgumentType]
        normalizer=normalizer,
        augmentations=Augmentations(),
    )

    num_trials, num_channels, num_steps = dataset.raw_emgs.shape
    num_patients = len(args.patient_ids)
    print(f"Trials: {num_trials}  Channels: {num_channels}  Time steps: {num_steps}")
    print(f"Windows: {len(dataset)}  Patients: {num_patients}")

    print("\n=== Dataset: data integrity ===")
    check("no NaN in raw_emgs", not torch.isnan(dataset.raw_emgs).any().item())
    check("no Inf in raw_emgs", not torch.isinf(dataset.raw_emgs).any().item())
    check(
        "labels aligned with trials",
        len(dataset.gestures) == num_trials and len(dataset.subjects) == num_trials,
        f"gestures={len(dataset.gestures)} subjects={len(dataset.subjects)} trials={num_trials}",
    )
    check(
        "patients present match requested",
        sorted(torch.unique(dataset.subjects).tolist()) == sorted(args.patient_ids),
    )
    check(
        "subject index is contiguous 0..N-1",
        sorted(dataset.subjects_to_idx.values()) == list(range(num_patients)),
    )

    sample = dataset.raw_emgs[
        torch.randperm(num_trials)[: min(256, num_trials)]
    ].float()
    print(
        f"  info: per-channel mean (sample) range "
        f"[{sample.mean(dim=(0, 2)).min():.3f}, {sample.mean(dim=(0, 2)).max():.3f}], "
        f"std range [{sample.std(dim=(0, 2)).min():.3f}, {sample.std(dim=(0, 2)).max():.3f}]"
    )
    del sample

    print("\n=== Dataset: window indices ===")
    wi = dataset.window_indices
    sizes_ok = all(int(w["end"]) - int(w["start"]) == args.window_size for w in wi)
    bounds_ok = all(0 <= int(w["start"]) and int(w["end"]) <= num_steps for w in wi)
    trials_ok = all(0 <= int(w["trial_idx"]) < num_trials for w in wi)
    check("every window has the requested size", sizes_ok)
    check("every window is within the signal", bounds_ok)
    check("every window points to a valid trial", trials_ok)

    print("\n=== Dataset: __getitem__ ===")
    idx = len(dataset) // 2
    window, window_aug, gesture, subject = dataset[idx]
    check(
        "window shape",
        tuple(window.shape) == (num_channels, args.window_size),
        str(tuple(window.shape)),
    )
    check("window dtype float32", window.dtype == torch.float32, str(window.dtype))
    check("aug is a tensor", isinstance(window_aug, torch.Tensor))
    if isinstance(window_aug, torch.Tensor):
        check(
            "aug shape matches window",
            window_aug.shape == window.shape,
            str(tuple(window_aug.shape)),
        )
        check("aug has no NaN/Inf", torch.isfinite(window_aug).all().item())
        diff = (window_aug - window).abs().mean().item()
        check("augmentation changes the data", diff > 1e-6, f"mean abs diff {diff:.6f}")
    trial_idx = int(wi[idx]["trial_idx"])
    check(
        "gesture label matches trial",
        torch.equal(gesture, dataset.gestures[trial_idx].long()),
    )
    check(
        "subject label matches trial",
        int(subject) == dataset.subjects_to_idx[int(dataset.subjects[trial_idx])],
        f"subject={int(subject)}",
    )
    check(
        "window content matches raw slice",
        torch.equal(
            window,
            dataset.raw_emgs[
                trial_idx, :, int(wi[idx]["start"]) : int(wi[idx]["end"])
            ].float(),
        )
        if dataset.rms_opts is None
        else True,
    )

    print("\n=== Dataset: no-augmentation path ===")
    plain = PhysioMioDataset(
        pathlib.Path(args.data),
        args.patient_ids[:1],
        window_opts=window_opts,  # pyright: ignore[reportArgumentType]
        augmentations=None,
    )
    _, aug_none, _, _ = plain[0]
    check(
        "aug view is a tensor when augmentations=None",
        isinstance(aug_none, torch.Tensor),
        f"got {type(aug_none).__name__} ({aug_none!r}); this is the `else 0` bug"
        if not isinstance(aug_none, torch.Tensor)
        else "",
    )
    del plain

    print("\n=== Sampler ===")
    sampler = PhysioMioTrialBatchSampler(window_indices=wi, seed=0)
    trial_counts = [len(w) for w in sampler.trial_windows]
    num_sampler_trials = len(sampler.trials)
    print(
        f"Trials in sampler: {num_sampler_trials}  windows/trial: min={min(trial_counts)} "
        f"max={max(trial_counts)}  used per epoch: {sampler.windows_per_trial}"
    )
    dropped = sum(c - sampler.windows_per_trial for c in trial_counts)
    print(
        f"  info: {dropped} windows ({dropped / len(wi):.1%}) never used per epoch due to min() truncation"
    )

    check(
        "sampler covers all dataset trials",
        num_sampler_trials == num_trials,
        f"{num_sampler_trials} vs {num_trials}",
    )
    check("__len__ == windows_per_trial", len(sampler) == sampler.windows_per_trial)

    epochs: list[list[list[int]]] = []
    for _ in range(args.epochs):
        epochs.append(list(iter(sampler)))

    first = epochs[0]
    check(
        "batches per epoch == len(sampler)", len(first) == len(sampler), f"{len(first)}"
    )
    check(
        "batch size == number of trials",
        all(len(b) == num_sampler_trials for b in first),
    )

    one_per_trial = True
    covers_all_trials = True
    for b in first:
        trials_in_batch = [int(wi[i]["trial_idx"]) for i in b]
        if len(set(trials_in_batch)) != len(trials_in_batch):
            one_per_trial = False
        if set(trials_in_batch) != set(sampler.trials):
            covers_all_trials = False
    check("exactly one window per trial in every batch", one_per_trial)
    check("every batch contains every trial", covers_all_trials)

    flat = [i for b in first for i in b]
    check("no window repeated within an epoch", len(flat) == len(set(flat)))

    b0 = first[0]
    trials0 = torch.tensor([int(wi[i]["trial_idx"]) for i in b0])
    n_gest = len(torch.unique(dataset.gestures[trials0]))
    n_subj = len(torch.unique(dataset.subjects[trials0]))
    check(
        "batch spans all patients", n_subj == num_patients, f"{n_subj}/{num_patients}"
    )
    print(f"  info: gestures per batch: {n_gest}")

    if args.epochs > 1 and sampler.windows_per_trial > 1:
        check("epochs are reshuffled (differ from each other)", epochs[0] != epochs[1])
    fresh = PhysioMioTrialBatchSampler(window_indices=wi, seed=0)
    check("same seed reproduces first epoch", list(iter(fresh)) == epochs[0])

    print("\n=== DataLoader with batch_sampler ===")
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    check("len(loader) == len(sampler)", len(loader) == len(sampler))

    emgs, emgs_aug, gestures, subjects = next(iter(loader))
    print(
        f"  info: emgs {tuple(emgs.shape)}, aug {tuple(emgs_aug.shape)}, "
        f"gestures {tuple(gestures.shape)}, subjects {tuple(subjects.shape)}"
    )
    check(
        "batch emg shape",
        tuple(emgs.shape) == (num_sampler_trials, num_channels, args.window_size),
    )
    check("batch aug shape matches emg", emgs_aug.shape == emgs.shape)
    check(
        "batch has no NaN/Inf",
        torch.isfinite(emgs).all().item() and torch.isfinite(emgs_aug).all().item(),
    )
    check(
        "gesture labels in range",
        int(gestures.min()) >= 0,
        f"unique: {torch.unique(gestures).tolist()}",
    )
    check(
        "subject labels in 0..N-1",
        int(subjects.min()) >= 0 and int(subjects.max()) < num_patients,
        f"min={int(subjects.min())} max={int(subjects.max())}",
    )

    print("\n=== Summary ===")
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
