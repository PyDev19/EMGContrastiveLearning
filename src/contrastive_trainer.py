from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

from src.config import TrainConfig
from src.loss import SupervisedContrastiveLoss


class BaseContrastiveTrainer(ABC):
    def __init__(
        self,
        model,
        optimizer: AdamW,
        scheduler: SequentialLR,
        loss_fn: SupervisedContrastiveLoss,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        config: TrainConfig,
        run: wandb.Run,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.compile(model.to(self.device))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = torch.compile(loss_fn)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.run = run

        self.pooled_indices = None
        self.pooled_embeddings: np.ndarray | None = None
        self.pooled_labels: np.ndarray | None = None
        self.pooled_subjects: np.ndarray | None = None

        self.epoch = 0

        self.grad_norm_sum = torch.zeros((), device=self.device)
        self.grad_norm_max = torch.zeros((), device=self.device)
        self.num_updates = 0

    def _stratified_subsample(
        self, h_arr: np.ndarray, labels_arr: np.ndarray, subjects_arr: np.ndarray
    ) -> None:
        if self.pooled_indices is None:
            n = len(h_arr)
            n_classes = len(np.unique(labels_arr))

            train_size = min(self.config.max_embedding_samples, n - n_classes)

            if train_size >= n_classes:
                self.pooled_indices, _ = train_test_split(
                    np.arange(len(h_arr)),
                    train_size=train_size,
                    stratify=labels_arr,
                    random_state=42,
                )
            else:
                self.pooled_indices = np.arange(n)

        self.pooled_embeddings = h_arr[self.pooled_indices]
        self.pooled_labels = labels_arr[self.pooled_indices]
        self.pooled_subjects = subjects_arr[self.pooled_indices]

    def _project_embeddings(self) -> dict:
        tsne = TSNE(n_components=3, random_state=42)
        tsne_coords = tsne.fit_transform(self.pooled_embeddings)

        umap = UMAP(n_components=3, random_state=42)
        umap_coords = umap.fit_transform(self.pooled_embeddings)

        gesture_tsne_points = np.concatenate(
            [tsne_coords, self.pooled_labels.reshape(-1, 1)], axis=1
        ).astype(np.float32)
        subject_tsne_points = np.concatenate(
            [tsne_coords, self.pooled_subjects.reshape(-1, 1)], axis=1
        ).astype(np.float32)

        gesture_umap_points = np.concatenate(  # pyright: ignore[reportCallIssue]
            [umap_coords, self.pooled_labels.reshape(-1, 1)],  # pyright: ignore[reportArgumentType]
            axis=1,
        ).astype(np.float32)
        subject_umap_points = np.concatenate(  # pyright: ignore[reportCallIssue]
            [umap_coords, self.pooled_subjects.reshape(-1, 1)],  # pyright: ignore[reportArgumentType]
            axis=1,
        ).astype(np.float32)

        return {
            "embeddings/gesture_tsne": wandb.Object3D(gesture_tsne_points),
            "embeddings/subject_tsne": wandb.Object3D(subject_tsne_points),
            "embeddings/gesture_umap": wandb.Object3D(gesture_umap_points),
            "embeddings/subject_umap": wandb.Object3D(subject_umap_points),
        }

    @torch.no_grad()
    def _linear_probe(self) -> dict:
        self.model.eval()
        train_h, train_labels, train_subjects = [], [], []

        for emg, _, labels, subjects in tqdm(
            self.train_dataloader,
            desc=f"[Epoch {self.epoch}] Extracting train embeddings",
        ):
            emg = emg.to(self.device)
            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                h, _ = self.model(emg, return_projected=False)

            train_h.append(h.float().cpu().numpy())
            train_labels.append(labels.numpy())
            train_subjects.append(subjects.numpy())

        train_h = np.concatenate(train_h, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_subjects = np.concatenate(train_subjects, axis=0)

        lr_classifier = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr_classifier.fit(train_h, train_labels)

        knn_classifier = KNeighborsClassifier(n_neighbors=20)
        knn_classifier.fit(train_h, train_labels)

        lr_preds = lr_classifier.predict(self.pooled_embeddings)
        knn_preds = knn_classifier.predict(self.pooled_embeddings)

        lr_report = classification_report(
            self.pooled_labels, lr_preds, output_dict=True
        )
        knn_report = classification_report(
            self.pooled_labels, knn_preds, output_dict=True
        )

        metrics: dict[str, float] = {}
        for name, report in [("logreg", lr_report), ("knn", knn_report)]:
            for label, m in report.items():  # pyright: ignore[reportAttributeAccessIssue]
                if label in ("accuracy", "macro avg", "weighted avg"):
                    continue
                metrics[f"linear_probe/{name}/f1_{label}"] = m["f1-score"]
            metrics[f"linear_probe/{name}/macro_f1"] = report["macro avg"]["f1-score"]  # pyright: ignore[reportArgumentType]

        lr_classifier.fit(train_h, train_subjects)
        knn_classifier.fit(train_h, train_subjects)

        lr_preds = lr_classifier.predict(self.pooled_embeddings)
        knn_preds = knn_classifier.predict(self.pooled_embeddings)

        lr_report = classification_report(
            self.pooled_subjects, lr_preds, output_dict=True
        )
        knn_report = classification_report(
            self.pooled_subjects, knn_preds, output_dict=True
        )

        for name, report in [("logreg", lr_report), ("knn", knn_report)]:
            for label, m in report.items():  # pyright: ignore[reportAttributeAccessIssue]
                if label in ("accuracy", "macro avg", "weighted avg"):
                    continue
                metrics[f"subject_probe/{name}/f1_{label}"] = m["f1-score"]
            metrics[f"subject_probe/{name}/macro_f1"] = report["macro avg"]["f1-score"]  # pyright: ignore[reportArgumentType]

        return metrics

    def _backward(self, loss: torch.Tensor, i: int) -> None:
        (loss / self.config.accumulation_steps).backward()
        is_last = (i + 1) == len(self.train_dataloader)
        if (i + 1) % self.config.accumulation_steps == 0 or is_last:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=float("inf")
            )
            norm = norm.detach()
            self.grad_norm_sum += norm
            self.grad_norm_max = torch.maximum(self.grad_norm_max, norm)
            self.num_updates += 1

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def _grad_metrics(self) -> dict[str, float]:
        n = max(self.num_updates, 1)
        return {
            "train/grad_norm_mean": (self.grad_norm_sum / n).item(),
            "train/grad_norm_max": self.grad_norm_max.item(),
        }

    @torch.no_grad
    def _calculate_similarity_stats(self, z: torch.Tensor, labels: torch.Tensor):
        z = torch.nn.functional.normalize(z.float(), dim=-1)
        labels.view(-1)
        sim_scores = z @ z.mT

        off_diag = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        same = (labels[:, None] == labels[None, :]) & off_diag
        diff = labels[:, None] != labels[None, :]

        mean_offdiag = sim_scores[off_diag].mean()
        mean_same = sim_scores[same].mean()
        mean_diff = sim_scores[diff].mean()
        mean_std = z.std(dim=0).mean()

        return mean_offdiag, mean_same, mean_diff, mean_std

    @abstractmethod
    def _train_epoch(self) -> dict[str, float]: ...

    @abstractmethod
    def _eval_epoch(self) -> dict[str, float]: ...

    def train(self):
        for epoch in range(1, self.config.num_epochs + 1):
            self.epoch = epoch

            self.run.log(self._train_epoch(), step=epoch)
            self.run.log(self._grad_metrics(), step=epoch)

            self.run.log({"train/lr": self.scheduler.get_last_lr()[0]}, step=epoch)

            self.run.log(self._eval_epoch(), step=epoch)

            if epoch % self.config.embeddings_log_freq == 0:
                self.run.log(self._project_embeddings(), step=epoch)

            if epoch % self.config.linear_probe_freq == 0:
                self.run.log(self._linear_probe(), step=epoch)

        self.run.finish()


class TimeContrastiveTrainer(BaseContrastiveTrainer):
    def _train_epoch(self):
        self.model.train()
        g_loss_sum = torch.zeros(1, device=self.device)
        num_samples = 0

        self.optimizer.zero_grad()
        for i, (emg, emg_aug, labels, _) in enumerate(
            tqdm(self.train_dataloader, desc=f"[Epoch {self.epoch}] Training")
        ):
            emg = emg.to(self.device)
            emg_aug = emg_aug.to(self.device)
            labels = labels.to(self.device)

            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                h, z = self.model(emg)
                _, z_aug = self.model(emg_aug)

            g_loss = self.loss_fn(z.float(), labels, z_aug=z_aug.float())
            self._backward(g_loss, i)

            batch_size = emg.size(0)
            g_loss_sum += g_loss.detach() * batch_size
            num_samples += batch_size

        proj_mean_offdiag, proj_mean_same, proj_mean_diff, proj_mean_std = (
            self._calculate_similarity_stats(z, labels)
        )
        pooled_mean_offdiag, pooled_mean_same, pooled_mean_diff, pooled_mean_std = (
            self._calculate_similarity_stats(h, labels)
        )

        return {
            "train/gesture_loss": (g_loss_sum / num_samples).item(),
            "train/proj_mean_offdiag": proj_mean_offdiag,
            "train/proj_mean_same": proj_mean_same,
            "train/proj_mean_diff": proj_mean_diff,
            "train/proj_mean_std": proj_mean_std,
            "train/pooled_mean_offdiag": pooled_mean_offdiag,
            "train/pooled_mean_same": pooled_mean_same,
            "train/pooled_mean_diff": pooled_mean_diff,
            "train/pooled_mean_std": pooled_mean_std,
        }

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        h_arr, labels_arr, subjects_arr = [], [], []
        loss_sum = torch.zeros(1, device=self.device)
        num_samples = 0

        for emg, _, labels, subjects in tqdm(
            self.test_dataloader, desc=f"[Epoch {self.epoch}] Evalutating"
        ):
            emg = emg.to(self.device)
            labels_gpu = labels.to(self.device)

            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                h, z = self.model(emg)

            loss = self.loss_fn(z.float(), labels_gpu)

            batch_size = emg.size(0)
            loss_sum += loss.detach() * batch_size
            num_samples += batch_size

            h_arr.append(h.float().cpu().numpy())
            labels_arr.append(labels.numpy())
            subjects_arr.append(subjects.numpy())

        proj_mean_offdiag, proj_mean_same, proj_mean_diff, proj_mean_std = (
            self._calculate_similarity_stats(z, labels_gpu)
        )
        pooled_mean_offdiag, pooled_mean_same, pooled_mean_diff, pooled_mean_std = (
            self._calculate_similarity_stats(h, labels_gpu)
        )

        h_arr = np.concatenate(h_arr, axis=0)
        labels_arr = np.concatenate(labels_arr, axis=0)
        subjects_arr = np.concatenate(subjects_arr, axis=0)

        gesture_silhouette_score = float(
            silhouette_score(h_arr, labels_arr, metric="cosine")
        )
        subject_silhouette_score = float(
            silhouette_score(h_arr, subjects_arr, metric="cosine")
        )

        self._stratified_subsample(h_arr, labels_arr, subjects_arr)

        return {
            "test/gesture_loss": (loss_sum / num_samples).item(),
            "test/gesture_silhouette_score": gesture_silhouette_score,
            "test/subject_silhouette_score": subject_silhouette_score,
            "test/proj_mean_offdiag": proj_mean_offdiag,
            "test/proj_mean_same": proj_mean_same,
            "test/proj_mean_diff": proj_mean_diff,
            "test/proj_mean_std": proj_mean_std,
            "test/pooled_mean_offdiag": pooled_mean_offdiag,
            "test/pooled_mean_same": pooled_mean_same,
            "test/pooled_mean_diff": pooled_mean_diff,
            "test/pooled_mean_std": pooled_mean_std,
        }
