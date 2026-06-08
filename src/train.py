import argparse
import pathlib

import torch
from plotly import express as px
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from umap import UMAP
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import silhouette_score, f1_score

from src.config import load_config
from src.datasets.physiomio import TFCDataset
from src.loss import TFCLoss
from src.utils import (
    build_model,
    load_fold_stats,
    prepare_run_dirs,
    save_fold_stats,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TFC model.")
    parser.add_argument("--data_path", type=str, help="Path to the training data.")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=1)
    return parser.parse_args()


def build_datasets(
    data_dir: pathlib.Path, train_patients: list, test_patients: list, fold: int, config
):
    fold_stats_path = data_dir / "fold_stats" / f"fold_{fold}.h5"

    if fold_stats_path.exists():
        print(f"Loading precomputed fold statistics for fold {fold}.")
        mean, std = load_fold_stats(fold_stats_path)
    else:
        print(f"Computing fold statistics for fold {fold} from training data.")
        mean, std = None, None

    train_dataset = TFCDataset(
        data_dir=data_dir,
        patient_ids=train_patients,
        mean=mean,
        std=std,
        **config.dataset,
    )

    if mean is None:
        save_fold_stats(fold_stats_path, train_dataset.mean, train_dataset.std)

    test_dataset = TFCDataset(
        data_dir=data_dir,
        patient_ids=test_patients,
        mean=train_dataset.mean,
        std=train_dataset.std,
        **config.dataset,
    )

    return train_dataset, test_dataset


def build_loaders(train_dataset, test_dataset, config):
    n_train = int(0.8 * len(train_dataset))
    train_dataset, val_dataset = random_split(
        train_dataset, [n_train, len(train_dataset) - n_train]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.val_batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.val_batch_size, shuffle=False, pin_memory=True
    )

    print(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader


def run_epoch(phase, loader, model, loss_fn, optimizer, grad_scaler, device):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    epoch_loss = 0.0
    epoch_time_loss = 0.0
    epoch_freq_loss = 0.0
    epoch_consistency_loss = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=phase.capitalize()):
            emg_window, fft_window, time_aug, freq_aug, _ = [
                x.to(device) for x in batch
            ]

            with torch.amp.autocast(device_type=device.type):
                ht, zt, hf, zf = model(emg_window, fft_window)
                ht_aug, zt_aug, hf_aug, zf_aug = model(time_aug, freq_aug)
                total_loss, time_loss, freq_loss, consistency_loss = loss_fn(
                    ht, ht_aug, hf, hf_aug, zt, zf, zt_aug, zf_aug
                )

            if is_train:
                optimizer.zero_grad()
                grad_scaler.scale(total_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            epoch_loss += total_loss.item()
            epoch_time_loss += time_loss
            epoch_freq_loss += freq_loss
            epoch_consistency_loss += consistency_loss

    n = len(loader)
    return (
        epoch_loss / n,
        epoch_time_loss / n,
        epoch_freq_loss / n,
        epoch_consistency_loss / n,
    )


def log_epoch(logger, phase, epoch, total, time_loss, freq_loss, consistency):
    logger.add_scalar(f"{phase}/loss", total, epoch)
    logger.add_scalar(f"{phase}/time_loss", time_loss, epoch)
    logger.add_scalar(f"{phase}/freq_loss", freq_loss, epoch)
    logger.add_scalar(f"{phase}/consistency_loss", consistency, epoch)


def extract_embeddings(
    loader: DataLoader, model: Module, device: torch.device, phase: str
):
    model.eval()
    all_zt = []
    all_zf = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{phase}] Extracting Embeddings", leave=False):
            emg, fft, _, _, labels = [x.to(device) for x in batch]
            _, z_time, _, z_freq = model(emg, fft)
            all_zt.append(z_time)
            all_zf.append(z_freq)
            all_labels.append(labels)

    all_zt = torch.cat(all_zt)
    all_zf = torch.cat(all_zf)
    all_labels = torch.cat(all_labels)

    return all_zt, all_zf, all_labels


def eval_linear_probe(model, train_loader, test_loader, device, log_dir, epoch):
    model.eval()
    all_zt_train, all_zf_train, all_labels_train = extract_embeddings(
        train_loader, model, device, phase="train"
    )
    all_zt_test, all_zf_test, all_labels_test = extract_embeddings(
        test_loader, model, device, phase="test"
    )

    z_train = torch.cat([all_zt_train, all_zf_train], dim=1)
    z_test = torch.cat([all_zt_test, all_zf_test], dim=1)

    mean = z_train.mean(dim=0, keepdim=True)
    std = z_train.std(dim=0, keepdim=True) + 1e-8
    z_train = (z_train - mean) / std
    z_test = (z_test - mean) / std

    labels_train = all_labels_train.cpu().numpy()
    labels_test = all_labels_test.cpu().numpy()

    silhouette = silhouette_score(z_test.cpu().numpy(), labels_test, metric="cosine")

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(z_train.cpu().numpy(), labels_train)

    logistic_reg = LogisticRegression(max_iter=1000)
    logistic_reg.fit(z_train.cpu().numpy(), labels_train)

    svm = LinearSVC(max_iter=10000)
    svm.fit(z_train.cpu().numpy(), labels_train)

    knn_f1 = f1_score(labels_test, knn.predict(z_test.cpu().numpy()), average="macro")
    logistic_f1 = f1_score(
        labels_test, logistic_reg.predict(z_test.cpu().numpy()), average="macro"
    )
    svm_f1 = f1_score(labels_test, svm.predict(z_test.cpu().numpy()), average="macro")

    print(f"Linear Probe Results at Epoch {epoch}:")
    print(f"  KNN F1 Score: {knn_f1:.4f}")
    print(f"  Logistic Regression F1 Score: {logistic_f1:.4f}")
    print(f"  SVM F1 Score: {svm_f1:.4f}")
    print(f"  Silhouette Score: {silhouette:.4f}")

    return knn_f1, logistic_f1, svm_f1, silhouette


def main():
    args = parse_args()
    config = load_config(args.config)

    logs_dir = prepare_run_dirs(pathlib.Path(args.log_dir))
    print(f"Logs and checkpoints: {logs_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(args.fold)

    train_patients = [p for p in range(1, 49) if p != args.fold]
    test_patients = [args.fold]
    print(f"Train patients: {train_patients} | Test patient: {test_patients}")

    train_dataset, test_dataset = build_datasets(
        pathlib.Path(args.data_path), train_patients, test_patients, args.fold, config
    )
    train_loader, val_loader, test_loader = build_loaders(
        train_dataset, test_dataset, config
    )

    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), **config.optimizer)
    loss_fn = TFCLoss(**config.loss)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    grad_scaler = torch.amp.GradScaler(device=device.type)
    logger = SummaryWriter(log_dir=logs_dir)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")

        for phase, loader in [
            ("train", train_loader),
            ("val", val_loader),
        ]:
            total, time_loss, freq_loss, consistency = run_epoch(
                phase, loader, model, loss_fn, optimizer, grad_scaler, device
            )
            log_epoch(logger, phase, epoch, total, time_loss, freq_loss, consistency)
            print(
                f"{phase:5s} — loss: {total:.4f} | time: {time_loss:.4f} | freq: {freq_loss:.4f} | consistency: {consistency:.4f}"
            )

            if phase == "val" and total < best_val_loss:
                best_val_loss = total
                best_epoch = epoch
                torch.save(model.state_dict(), logs_dir / "best_model.pt")
                print(
                    f"New best model saved (epoch {epoch}, val loss {best_val_loss:.4f})"
                )

        if epoch % 10 == 0:
            knn_f1, logistic_f1, svm_f1, silhouette = eval_linear_probe(
                model, train_loader, test_loader, device, logs_dir, epoch
            )
            logger.add_scalar("linear_probe/knn_f1", knn_f1, epoch)
            logger.add_scalar("linear_probe/logistic_f1", logistic_f1, epoch)
            logger.add_scalar("linear_probe/svm_f1", svm_f1, epoch)
            logger.add_scalar("linear_probe/silhouette", silhouette, epoch)

        scheduler.step()

    print(
        f"\nTraining complete. Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}"
    )
    logger.close()


if __name__ == "__main__":
    main()
