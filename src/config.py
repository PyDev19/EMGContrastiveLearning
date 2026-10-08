import pathlib
from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.datasets import *
from src.loss import *
from src.models import *
from src.utils.augmentations import Augmentations
from src.utils.registers import DATASETS, LOSSES, MODELS, NORMALIZERS
from src.utils.types import (
    ActivationName,
    DatasetName,
    NormalizerName,
    WindowOpts,
)


@dataclass
class DatasetArgs:
    window_opts: WindowOpts
    rms_opts: WindowOpts | None = None


@dataclass
class DatasetConfig:
    name: DatasetName
    args: DatasetArgs


@dataclass
class RoFormerContrastiveModelArgs:
    time_steps: int
    channels: int
    patch_size: int
    embed_dim: int
    hidden_dim: int
    projection_dim: int
    projection_hidden_dims: list[int]
    num_heads: int
    num_layers: int
    proj_drop_prob: float
    attn_drop_prob: float
    drop_path_prob: float
    mlp_drop_prob: float
    qkv_bias: bool
    projection_activation: ActivationName


@dataclass
class SupervisedContrastiveLossArgs:
    temperature: float


@dataclass
class OptimizerArgs:
    learning_rate: float
    weight_decay: float


@dataclass
class WandbConfig:
    entity: str
    project: str
    log_freq: int


@dataclass
class TimeContrastiveModelConfig:
    name: Literal["roformer_time_contrastive"]
    args: RoFormerContrastiveModelArgs


@dataclass
class TimeDomainAdversarialModelConfig:
    name: Literal["roformer_time_domain_adversial_contrastive"]
    args: RoFormerContrastiveModelArgs


@dataclass
class TimeContrastiveTrainingConfig:
    task: Literal["time_contrastive"]
    batch_size: int
    num_epochs: int
    max_embedding_samples: int
    embeddings_log_freq: int
    linear_probe_freq: int

    normalizer: NormalizerName
    dataset: DatasetConfig
    model: TimeContrastiveModelConfig
    loss: SupervisedContrastiveLossArgs
    optimizer: OptimizerArgs
    wandb: WandbConfig


@dataclass
class TimeDomainAdversarialTrainingConfig:
    task: Literal["time_domain_adversial_contrastive"]
    batch_size: int
    num_epochs: int
    max_embedding_samples: int
    embeddings_log_freq: int
    linear_probe_freq: int

    normalizer: NormalizerName
    dataset: DatasetConfig
    model: TimeDomainAdversarialModelConfig
    loss: SupervisedContrastiveLossArgs
    optimizer: OptimizerArgs
    wandb: WandbConfig


TrainingConfig = TimeContrastiveTrainingConfig | TimeDomainAdversarialTrainingConfig

_TASK_SCHEMA_MAP: dict[str, type] = {
    "time_contrastive": TimeContrastiveTrainingConfig,
    "time_domain_adversial_contrastive": TimeDomainAdversarialTrainingConfig,
}


def load_training_config(config_path: str) -> TrainingConfig:
    """Load and validate a training config from YAML, dispatching to the schema
    matching the top-level `task` field. Because each task variant's `model`
    field is restricted to a Literal of only that task's valid model name(s),
    a mismatched task/model pairing fails OmegaConf's structured-config
    validation directly — no separate runtime cross-check needed.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        TrainingConfig: the validated, fully-typed training config.

    Raises:
        ValueError: if `task` is missing or not a recognized task name.
    """
    raw = OmegaConf.load(config_path)

    task = OmegaConf.select(raw, "task")
    if task not in _TASK_SCHEMA_MAP:
        raise ValueError(
            f"Unknown or missing 'task': {task!r}. Expected one of {list(_TASK_SCHEMA_MAP)}."
        )

    schema = _TASK_SCHEMA_MAP[task]
    merged = OmegaConf.merge(OmegaConf.structured(schema), raw)
    return cast(TrainingConfig, OmegaConf.to_object(merged))


def build_dataloaders(config: TrainingConfig, data_dir: pathlib.Path):
    normalizer = NORMALIZERS[config.normalizer]()
    augmentations = Augmentations()

    if config.dataset.name == "physiomio":
        patients = list(range(1, 49))
        test_patient = np.random.choice(patients, size=5, replace=False).tolist()
        train_patients = [p for p in patients if p not in test_patient]
    else:
        raise NotImplementedError(
            f"Patient splitting for '{config.dataset.name}' not yet implemented"
        )

    print(f"Train patients: {train_patients}")
    print(f"Test patients: {test_patient}")

    train_dataset = DATASETS[config.dataset.name](
        data_dir=data_dir,
        patient_ids=train_patients,
        normalizer=normalizer,
        augmentations=augmentations,
        **asdict(config.dataset.args),
    )

    test_dataset = DATASETS[config.dataset.name](
        data_dir=data_dir,
        patient_ids=test_patient,
        normalizer=normalizer,
        augmentations=augmentations,
        **asdict(config.dataset.args),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


def build_model_optimizer_scheduler(config: TrainingConfig):
    match config:
        case TimeContrastiveTrainingConfig():
            model = MODELS[config.model.name](**asdict(config.model.args))
        case TimeDomainAdversarialTrainingConfig():
            model = MODELS[config.model.name](**asdict(config.model.args))
        case _:
            raise ValueError(f"Unhandled training config variant: {type(config)}")

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    return model, optimizer, scheduler


def build_loss(config: TrainingConfig):
    match config:
        case TimeContrastiveTrainingConfig():
            return LOSSES["supervised_contrastive"](**asdict(config.loss))
        case TimeDomainAdversarialTrainingConfig():
            return LOSSES["supervised_contrastive"](**asdict(config.loss))
        case _:
            raise ValueError(f"Unhandled training config variant: {type(config)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate training config YAML.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    try:
        config = load_training_config(args.config)
        print("Config loaded and validated successfully:")
        print(config)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading config: {e}")
