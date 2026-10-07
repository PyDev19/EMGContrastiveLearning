from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.amp import GradScaler, autocast
from umap import UMAP


class BaseContrastiveTrainer(ABC):
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_dataloader,
        test_dataloader,
        config,
        run,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.run = run

        self.pooled_indices = None
        self.pooled_embeddings: np.ndarray | None = None
        self.pooled_labels: np.ndarray | None = None

    def _stratified_subsample(self, h_arr: np.ndarray, labels_arr: np.ndarray) -> None:
        if self.pooled_indices is None:
            self.pooled_indices, _ = train_test_split(
                np.arange(len(h_arr)),
                train_size=self.config.max_embedding_samples,
                stratify=labels_arr,
                random_state=42,
            )

        self.pooled_embeddings = h_arr[self.pooled_indices]
        self.pooled_labels = labels_arr[self.pooled_indices]

    def _project_embeddings(self) -> dict:
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        tsne_coords = tsne.fit_transform(self.pooled_embeddings)

        umap = UMAP(n_components=3, random_state=42, n_jobs=-1)
        umap_coords = umap.fit_transform(self.pooled_embeddings)

        gesture_tsne_points = np.concatenate(
            [tsne_coords, self.pooled_labels.reshape(-1, 1)], axis=1
        ).astype(np.float32)
        gesture_umap_points = np.concatenate(  # pyright: ignore[reportCallIssue]
            [umap_coords, self.pooled_labels.reshape(-1, 1)],  # pyright: ignore[reportArgumentType]
            axis=1,
        ).astype(np.float32)

        return {
            "embeddings/gesture_tsne": wandb.Object3D(gesture_tsne_points),
            "embeddings/gesture_umap": wandb.Object3D(gesture_umap_points),
        }

    @torch.no_grad()
    def linear_probe(self) -> dict:
        self.model.eval()
        train_h, train_labels = [], []

        for emg, _, labels, _ in self.train_dataloader:
            emg = emg.to(self.device)
            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                h, _ = self.model(emg, return_projected=False)
            train_h.append(h.float().cpu().numpy())
            train_labels.append(labels.numpy())

        train_h = np.concatenate(train_h, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

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

        def extract_f1(report: dict) -> dict[str, float]:
            return {
                f"label_{label}": m["f1-score"]
                for label, m in report.items()
                if label not in ("accuracy", "macro avg", "weighted avg")
            }

        return {
            "linear_probe/logreg": extract_f1(lr_report),  # pyright: ignore[reportArgumentType]
            "linear_probe/knn": extract_f1(knn_report),  # pyright: ignore[reportArgumentType]
        }

    @abstractmethod
    def _train_epoch(self): ...

    @abstractmethod
    def _eval_epoch(self): ...

    def train(self):
        for epoch in range(1, self.config.num_epochs + 1):
            self.run.log(self._train_epoch(), step=epoch)
            self.scheduler.step()
            self.run.log({"train/lr": self.scheduler.get_last_lr()[0]}, step=epoch)

            self.run.log(self._eval_epoch(), step=epoch)

            if epoch % self.config.embeddings_log_freq == 0:
                self.run.log(self._project_embeddings(), step=epoch)

            if epoch % self.config.linear_probe_freq == 0:
                self.run.log(self.linear_probe(), step=epoch)

        self.run.finish()


class TimeContrastiveTrainer(BaseContrastiveTrainer):
    def _train_epoch(self) -> dict:
        self.model.train()
        g_loss_sum = torch.zeros(1, device=self.device)
        num_samples = 0

        for emg, emg_aug, labels, _ in self.train_dataloader:
            emg = emg.to(self.device)
            emg_aug = emg_aug.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                _, z = self.model(emg)
                _, z_aug = self.model(emg_aug)

            g_loss = self.loss_fn(z.float(), labels, z_aug=z_aug.float())

            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = emg.size(0)
            g_loss_sum += g_loss.detach() * batch_size
            num_samples += batch_size

        return {"train/gesture_loss": (g_loss_sum / num_samples).item()}

    @torch.no_grad()
    def _eval_epoch(self) -> dict:
        self.model.eval()
        h_arr, labels_arr = [], []
        loss_sum = torch.zeros(1, device=self.device)
        num_samples = 0

        for emg, _, labels, _ in self.test_dataloader:
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

        h_arr = np.concatenate(h_arr, axis=0)
        labels_arr = np.concatenate(labels_arr, axis=0)

        gesture_silhouette_score = float(silhouette_score(h_arr, labels_arr, metric="cosine"))
        self._stratified_subsample(h_arr, labels_arr)

        return {
            "test/gesture_loss": (loss_sum / num_samples).item(),
            "test/gesture_silhouette_score": gesture_silhouette_score,
        }
