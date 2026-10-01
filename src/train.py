import argparse
import pathlib
from dataclasses import asdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from umap import UMAP

import wandb
from src.config import ContrastiveTrainingConfig, load_training_config
from src.datasets.physiomio import PhysioMioDataset
from src.loss.supcon import SupervisedContrastiveLoss
from src.models.roformer_contrastive import RoFormerContrastiveModel
from src.utils.augmentations import Augmentations
from src.utils.registers import NORMALIZERS

DATASETS = {"physiomio": PhysioMioDataset}
MODELS = {"roformer_contrastive": RoFormerContrastiveModel}
LOSSES = {
    "supervised_contrastive": SupervisedContrastiveLoss,
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
):
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

    total_loss_sum = torch.zeros(1, device=device)
    gesture_loss_sum = torch.zeros(1, device=device)
    subject_loss_sum = torch.zeros(1, device=device)

    num_samples = 0

    for emg, emg_aug, labels, subjects in loader:
        emg = emg.to(device)
        emg_aug = emg_aug.to(device)
        labels = labels.to(device)
        subjects = subjects.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            _, z, z_subject = model(emg)
            _, z_aug, z_aug_subject = model(emg_aug)

        gesture_loss = loss_fn(z, labels, z_aug=z_aug)
        subject_loss = loss_fn(z_subject, subjects, z_aug=z_aug_subject)
        total_loss = gesture_loss + subject_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = emg.size(0)
        total_loss_sum += total_loss.detach() * batch_size
        gesture_loss_sum += gesture_loss.detach() * batch_size
        subject_loss_sum += subject_loss.detach() * batch_size
        num_samples += batch_size

    return {
        "total_loss": (total_loss_sum / num_samples).item(),
        "gesture_loss": (gesture_loss_sum / num_samples).item(),
        "subject_loss": (subject_loss_sum / num_samples).item(),
    }


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

    total_loss_sum = torch.zeros(1, device=device)
    gesture_loss_sum = torch.zeros(1, device=device)
    subject_loss_sum = torch.zeros(1, device=device)

    num_samples = 0
    all_pooled = []
    all_labels = []
    all_subjects = []

    for emg, _, labels, subjects in loader:
        emg = emg.to(device)
        labels = labels.to(device)
        subjects = subjects.to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            h, z, z_subject = model(emg)

        gesture_loss = loss_fn(z, labels)
        subject_loss = loss_fn(z_subject, subjects)
        total_loss = gesture_loss + subject_loss

        batch_size = emg.size(0)
        total_loss_sum += total_loss.detach() * batch_size
        gesture_loss_sum += gesture_loss.detach() * batch_size
        subject_loss_sum += subject_loss.detach() * batch_size
        num_samples += batch_size

        all_pooled.append(h.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_subjects.append(subjects.cpu().numpy())

    pooled = np.concatenate(all_pooled, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)

    gesture_silhouette_score = float(silhouette_score(pooled, labels, metric="cosine"))
    subject_silhouette_score = float(
        silhouette_score(pooled, subjects, metric="cosine")
    )

    return {
        "total_loss": (total_loss_sum / num_samples).item(),
        "gesture_loss": (gesture_loss_sum / num_samples).item(),
        "subject_loss": (subject_loss_sum / num_samples).item(),
        "gesture_silhouette_score": gesture_silhouette_score,
        "subject_silhouette_score": subject_silhouette_score,
        "embeddings": pooled,
        "labels": labels,
        "subjects": subjects,
    }


def log_embeddings(pooled, labels, subjects):
    """Logs embeddings to wandb.

    Args:
        run: wandb run object.
        pooled: pooled embeddings (pre-projection) of shape (N, D).
        labels: corresponding labels of shape (N,).
        step: current training step or epoch.
    """
    tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
    tsne_coords = tsne.fit_transform(pooled)

    umap = UMAP(n_components=3, random_state=42, n_jobs=-1)
    umap_coords = umap.fit_transform(pooled)

    gesture_tsne_points = np.concatenate([tsne_coords, labels.reshape(-1, 1)], axis=1)
    subject_tsne_points = np.concatenate([tsne_coords, subjects.reshape(-1, 1)], axis=1)

    gesture_umap_points = np.concatenate(
        [umap_coords, labels.reshape(-1, 1)],  # type: ignore
        axis=1,
    )
    subject_umap_points = np.concatenate(
        [umap_coords, subjects.reshape(-1, 1)],  # type: ignore
        axis=1,
    )

    return {
        "gesture_tsne": wandb.Object3D(gesture_tsne_points),
        "subject_tsne": wandb.Object3D(subject_tsne_points),
        "gesture_umap": wandb.Object3D(gesture_umap_points),
        "subject_umap": wandb.Object3D(subject_umap_points),
    }


@torch.no_grad()
def linear_probe(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_h: np.ndarray,
    test_labels: np.ndarray,
    test_subjects: np.ndarray,
    device: torch.device,
):
    """Trains a linear classifier on top of the frozen model's pooled embeddings and evaluates it.

    Args:
        model (torch.nn.Module): the contrastive model; forward returns (pooled, projected).
        train_loader (DataLoader): yields (emg, emg_aug, labels) batches for training.
        test_loader (DataLoader): yields (emg, emg_aug, labels) batches for evaluation.
        device (torch.device): device to move batches to.
    """
    model.eval()

    train_h, train_labels, train_subjects = [], [], []

    for emg, _, labels, subjects in train_loader:
        emg = emg.to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            h, _, _ = model(emg)

        train_h.append(h.cpu().numpy())
        train_labels.append(labels.cpu().numpy())
        train_subjects.append(subjects.cpu().numpy())

    train_h = np.concatenate(train_h, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_subjects = np.concatenate(train_subjects, axis=0)

    lr_classifier = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
    lr_classifier.fit(train_h, train_labels)

    knn_classifier = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    knn_classifier.fit(train_h, train_labels)

    lr_preds = lr_classifier.predict(test_h)
    knn_preds = knn_classifier.predict(test_h)

    lr_gesture_report = classification_report(test_labels, lr_preds, output_dict=True)
    knn_gesture_report = classification_report(test_labels, knn_preds, output_dict=True)

    lr_metrics = {}
    knn_metrics = {}

    for label, metrics in lr_gesture_report.items():  # type: ignore
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            lr_metrics[f"label_{label}"] = metrics["f1-score"]

    for label, metrics in knn_gesture_report.items():  # type: ignore
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            knn_metrics[f"label_{label}"] = metrics["f1-score"]

    lr_classifier.fit(train_h, train_subjects)
    knn_classifier.fit(train_h, train_subjects)

    lr_preds = lr_classifier.predict(test_h)
    knn_preds = knn_classifier.predict(test_h)

    lr_metrics["subjects"] = f1_score(test_subjects, lr_preds)
    knn_metrics["subjects"] = f1_score(test_subjects, knn_preds)

    return {"lr": lr_metrics, "knn": knn_metrics}


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
    scheduler = CosineAnnealingWarmRestarts(optimizer, **asdict(config.scheduler))
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
        train_metrics = run_train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )

        for key, value in train_metrics.items():
            run.log({f"train/{key}": value}, step=epoch)

        scheduler.step()
        run.log({"learning_rate": scheduler.get_last_lr()[0]}, step=epoch)

        eval_metrics = run_eval_epoch(model, test_loader, loss_fn, device)

        for key, value in eval_metrics.items():
            if key not in ["embeddings", "labels", "subjects"]:
                run.log({f"test/{key}": value}, step=epoch)

        if (
            pooled_indices is None
            and len(eval_metrics["embeddings"]) > config.max_embedding_samples
        ):
            pooled_indices, _ = train_test_split(
                np.arange(len(eval_metrics["embeddings"])),
                train_size=config.max_embedding_samples,
                stratify=eval_metrics["labels"],
                random_state=42,
            )

        if epoch % config.embeddings_log_freq == 0 and pooled_indices is not None:
            embedding_coords = log_embeddings(
                eval_metrics["embeddings"][pooled_indices],
                eval_metrics["labels"][pooled_indices],
                eval_metrics["subjects"][pooled_indices],
            )

            for key, value in embedding_coords.items():
                run.log({f"test_embeddings/{key}": value}, step=epoch)

        if epoch % config.linear_probe_freq == 0 and pooled_indices is not None:
            probe_results = linear_probe(
                model,
                train_loader,
                eval_metrics["embeddings"],
                eval_metrics["labels"],
                eval_metrics["subjects"],
                device,
            )

            for probe_name, metrics in probe_results.items():
                for label, f1_score in metrics.items():
                    run.log(
                        {f"probe_{probe_name}_f1/{label}": f1_score},
                        step=epoch,
                    )

    run.finish()


if __name__ == "__main__":
    main()
