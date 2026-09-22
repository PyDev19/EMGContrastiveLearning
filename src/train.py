import argparse
import pathlib
from dataclasses import asdict

import numpy as np
import torch
import wandb
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from umap import UMAP

from src.config import ContrastiveTrainingConfig, load_training_config
from src.datasets.physiomio import PhysioMioDataset
from src.loss.masked_reconstruction import MaskedReconstructionLoss
from src.loss.supcon import SupervisedContrastiveLoss
from src.models.roformer_contrastive import RoFormerContrastiveModel
from src.utils.augmentations import Augmentations
from src.utils.registers import NORMALIZERS

DATASETS = {"physiomio": PhysioMioDataset}
MODELS = {"roformer_contrastive": RoFormerContrastiveModel}
LOSSES = {
    "supervised_contrastive": SupervisedContrastiveLoss,
    "masked_reconstruction": MaskedReconstructionLoss,
}


def build_datasets(config: ContrastiveTrainingConfig, data_dir: pathlib.Path):
    normalizer = NORMALIZERS[config.normalizer]()
    augmentations = Augmentations()

    patients = list(range(1, 49))
    test_patient = np.random.choice(patients, size=5, replace=False).tolist()
    train_patients = [p for p in patients if p not in test_patient]

    dataset_args = config.dataset.args

    train_dataset = DATASETS[config.dataset.name](
        data_dir=data_dir,
        patient_ids=train_patients,
        normalizer=normalizer,
        window_opts=dataset_args.window_opts,
        rms_opts=dataset_args.rms_opts,
        augmentations=augmentations,
    )
    test_dataset = DATASETS[config.dataset.name](
        data_dir=data_dir,
        patient_ids=test_patient,
        normalizer=normalizer,
        window_opts=dataset_args.window_opts,
        rms_opts=dataset_args.rms_opts,
        augmentations=None,
    )
    return train_dataset, test_dataset


def build_model(config: ContrastiveTrainingConfig):
    return MODELS[config.model.name](**asdict(config.model.args))


def build_loss(config: ContrastiveTrainingConfig):
    return LOSSES[config.loss.name](**asdict(config.loss.args))


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: GradScaler,
) -> float:
    """Runs one training epoch of contrastive learning.

    Args:
        model (torch.nn.Module): the contrastive model; forward returns (pooled, projected).
        loader (DataLoader): yields (emg, emg_aug, labels) batches.
        loss_fn (torch.nn.Module): contrastive loss, called as loss_fn(z, z_aug, labels).
        optimizer (Optimizer): optimizer stepped once per batch.
        device (torch.device): device to move batches to.

    Returns:
        float: sample-weighted mean training loss across the epoch.
    """
    model.train()

    loss_sum = torch.zeros(1, device=device)
    num_samples = 0

    for emg, emg_aug, labels in loader:
        emg = emg.to(device)
        emg_aug = emg_aug.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            _, z = model(emg)
            _, z_aug = model(emg_aug)
            loss = loss_fn(z, labels, z_aug=z_aug)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = emg.size(0)
        loss_sum += loss.detach() * batch_size  # stays on GPU, no sync per-batch
        num_samples += batch_size

    return (loss_sum / num_samples).item()  # single sync at epoch end


@torch.no_grad()
def run_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Runs one evaluation epoch: computes mean loss and silhouette score on the
    pooled (pre-projection) backbone representation, not the projected embedding,
    since the pooled output better reflects general-purpose representation quality.

    Args:
        model (torch.nn.Module): the contrastive model; forward returns (pooled, projected).
        loader (DataLoader): yields (emg, emg_aug, labels) batches.
        loss_fn (torch.nn.Module): contrastive loss, called as loss_fn(z, z_aug, labels).
        device (torch.device): device to move batches to.

    Returns:
        dict[str, float]: {"loss": sample-weighted mean loss, "silhouette_score": ...}
            computed over both views' pooled embeddings across the full loader.
    """
    model.eval()

    loss_sum = torch.zeros(1, device=device)
    num_samples = 0
    all_pooled: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for emg, _, labels in loader:
        emg = emg.to(device)
        labels = labels.to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            h, z = model(emg)
            loss = loss_fn(z, labels)

        batch_size = emg.size(0)
        loss_sum += loss.detach() * batch_size
        num_samples += batch_size

        all_pooled.append(h.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    pooled = np.concatenate(all_pooled, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    test_silhouette_score = float(silhouette_score(pooled, labels, metric="cosine"))

    norms = np.linalg.norm(pooled, axis=1)

    return {
        "loss": (loss_sum / num_samples).item(),
        "silhouette_score": test_silhouette_score,
        "pooled_norms_mean": float(norms.mean()),
        "pooled_norms_std": float(norms.std()),
        "embeddings": pooled,
        "labels": labels,
    }


def log_embeddings(run, pooled, labels, step):
    """Logs embeddings to wandb.

    Args:
        run: wandb run object.
        pooled: pooled embeddings (pre-projection) of shape (N, D).
        labels: corresponding labels of shape (N,).
        step: current training step or epoch.
    """
    tsne = TSNE(n_components=3, random_state=42)
    tsne_coords = tsne.fit_transform(pooled)

    umap = UMAP(n_components=3, random_state=42)
    umap_coords = umap.fit_transform(pooled)

    tsne_points = np.concatenate([tsne_coords, labels.reshape(-1, 1)], axis=1)
    umap_points = np.concatenate([umap_coords, labels.reshape(-1, 1)], axis=1)  # type: ignore

    run.log(
        {
            "tsne_embeddings": wandb.Object3D(tsne_points),
            "umap_embeddings": wandb.Object3D(umap_points),
        },
        step=step,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    args = parser.parse_args()

    config = load_training_config(str(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset, test_dataset = build_datasets(config, args.data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_restarts)
    loss_fn = build_loss(config)

    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config={**asdict(config)},
    )
    run.watch(model, loss_fn, log="all", log_freq=config.wandb.log_freq)

    pooled_indices = None

    scaler = GradScaler(device=device.type)

    for epoch in range(1, config.num_epochs + 1):
        train_loss = run_train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )
        scheduler.step()
        eval_metrics = run_eval_epoch(model, test_loader, loss_fn, device)

        if (
            pooled_indices is None
            and len(eval_metrics["embeddings"]) > config.max_embedding_samples
        ):
            _, pooled_indices = train_test_split(
                np.arange(len(eval_metrics["embeddings"])),
                train_size=config.max_embedding_samples,
                stratify=eval_metrics["labels"],
                random_state=42,
            )

        run.log(
            {
                "train_loss": train_loss,
                "test_loss": eval_metrics["loss"],
                "test_silhouette": eval_metrics["silhouette_score"],
                "test_pooled_norms_mean": eval_metrics["pooled_norms_mean"],
                "test_pooled_norms_std": eval_metrics["pooled_norms_std"],
            },
            step=epoch,
        )

        if epoch % config.embeddings_log_freq == 0 and pooled_indices is not None:
            log_embeddings(
                run,
                eval_metrics["embeddings"][pooled_indices],
                eval_metrics["labels"][pooled_indices],
                step=epoch,
            )

    run.finish()


if __name__ == "__main__":
    main()
