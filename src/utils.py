import os
import pathlib
import random
import re

import h5py
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.nn import Module
from src.models.transformer_cls_tfc import CLSTransformerTFC

MODEL_REGISTRY = {
    "transformer_cls": CLSTransformerTFC,
}


class Metrics:
    @staticmethod
    def compute(preds: np.ndarray, labels: np.ndarray):
        return {
            "accuracy": accuracy_score(labels, preds),
            "recall": recall_score(labels, preds, average="macro"),
            "f1_score": f1_score(labels, preds, average="macro"),
        }


class EarlyStopping:
    def __init__(self, warmup: int = 10, patience: int = 10, min_delta: float = 1e-4):
        self.warmup = warmup
        self.inital_warmup = warmup
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, loss: float) -> bool:
        if self.warmup > 0:
            self.warmup -= 1
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.should_stop = False
        self.warmup = self.inital_warmup


def prepare_run_dirs(logs_dir: pathlib.Path) -> pathlib.Path:
    existing_runs = []
    if logs_dir.exists():
        for item in os.listdir(logs_dir):
            match = re.match(r"^run_(\d+)$", item)
            if match:
                existing_runs.append(int(match.group(1)))

    next_run = max(existing_runs) + 1 if existing_runs else 1
    run_dir = logs_dir / f"run_{next_run}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "scalers").mkdir(parents=True, exist_ok=True)
    (run_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_model(config) -> Module:
    model_cls = MODEL_REGISTRY.get(config.model.name)
    if model_cls is None:
        raise ValueError(
            f"Unknown model name {config.model.name!r} in config. Expected one of: {list(MODEL_REGISTRY)}"
        )

    model_kwargs = OmegaConf.to_container(config.model, resolve=True)
    model_kwargs.pop("name")

    return model_cls(**model_kwargs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model: torch.nn.Module, path: pathlib.Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **kwargs}, path)

def load_fold_stats(fold_stats_path: pathlib.Path):
    with h5py.File(fold_stats_path, "r") as f:
        return f["mean"][:], f["std"][:]


def save_fold_stats(fold_stats_path: pathlib.Path, mean, std):
    fold_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fold_stats_path, "w") as f:
        f.create_dataset("mean", data=mean)
        f.create_dataset("std", data=std)
