import pathlib
from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from src.datasets import *
from src.loss import *
from src.models import *
from src.utils.augmentations import Augmentations
from src.utils.registers import DATASETS, LOSSES, MODELS, NORMALIZERS, SAMPLERS
from src.utils.types import (
    DatasetName,
    LossName,
    ModelName,
    NormalizerName,
    NormalizerScopeName,
    TaskName,
    WindowOpts,
)


@dataclass
class AugmentationConfig:
    jitter_sigma: float
    scale_sigma: float
    mask_size: list[int]
    mask_prob: float
    channel_dropout_prob: float


@dataclass
class DatasetArgs:
    window_opts: WindowOpts
    rms_opts: WindowOpts | None


@dataclass
class DatasetConfig:
    name: DatasetName
    args: DatasetArgs


@dataclass
class ModelConfig:
    name: ModelName
    args: dict


@dataclass
class LossConfig:
    name: LossName
    args: dict


@dataclass
class WandbConfig:
    entity: str
    project: str
    log_freq: int


@dataclass
class TrainConfig:
    task: TaskName

    batch_size: int | Literal["sampler"]
    accumulation_steps: int
    num_epochs: int

    warmup: Literal["warmup", "rampup"]
    learning_rate: float
    weight_decay: float

    normalizer: NormalizerName
    normalizer_scope: NormalizerScopeName

    max_embedding_samples: int
    embeddings_log_freq: int
    linear_probe_freq: int

    augmentations: AugmentationConfig
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    wandb: WandbConfig


TASK_TO_MODEL: dict[TaskName, ModelName] = {
    "time_contrastive": "roformer_time_contrastive",
    "time_dann_contrastive": "roformer_time_dann_contrastive",
}

TASK_TO_LOSS: dict[TaskName, LossName] = {
    "time_contrastive": "supervised_contrastive",
    "time_dann_contrastive": "supervised_contrastive",
}

DATASET_TO_PATINETS: dict[DatasetName, list[int]] = {"physiomio": list(range(1, 49))}


def load_train_config(
    config_path: str, overrides: list[str] | dict[str, any] | None = None
) -> TrainConfig:
    """Load a training config from YAML and validate it against `TrainConfig`.

    Args:
        config_path (str): path to the YAML file.
        overrides (list[str] | None): dotlist overrides, e.g.
            ["learning_rate=3e-4", "model.args.num_layers=6"].

    Returns:
        TrainConfig: the fully-typed config. `model.args` and `loss.args` are plain dicts.

    Raises:
        omegaconf.errors.MissingMandatoryValue: a required field is absent from the YAML.
        omegaconf.errors.ValidationError: a field has the wrong type.
    """
    sources = [OmegaConf.structured(TrainConfig), OmegaConf.load(config_path)]
    if isinstance(overrides, dict):
        sources.append(
            OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()])
        )
    elif overrides:
        sources.append(OmegaConf.from_dotlist(overrides))

    merged = OmegaConf.merge(*sources)
    config = cast(TrainConfig, OmegaConf.to_object(merged))

    if not (
        config.model.name == TASK_TO_MODEL[config.task]
        and config.loss.name == TASK_TO_LOSS[config.task]
    ):
        raise ValueError(
            f"task '{config.task}' expects '{config.model.name}' model and '{config.loss.name}' loss"
        )

    return config


def build_dataloaders(
    config: TrainConfig, data_dir: pathlib.Path
) -> tuple[DataLoader, DataLoader, list[int]]:
    patients = DATASET_TO_PATINETS.get(config.dataset.name, [])
    if not patients:
        raise NotImplementedError(f"{config.dataset.name} not implemented")

    test_patients = np.random.choice(patients, size=5, replace=False).tolist()
    train_patients = [p for p in patients if p not in test_patients]

    augmentations = Augmentations(**asdict(config.augmentations))

    if config.normalizer_scope == "global":
        train_dataset = DATASETS[config.dataset.name](
            data_dir=data_dir,
            patient_ids=train_patients,
            normalizer=NORMALIZERS[config.normalizer](),
            augmentations=augmentations,
            **asdict(config.dataset.args),
        )

        test_dataset = DATASETS[config.dataset.name](
            data_dir=data_dir,
            patient_ids=test_patients,
            normalizer=train_dataset.normalizer,
            augmentations=augmentations,
            **asdict(config.dataset.args),
        )
    else:
        train_dataset = DATASETS[config.dataset.name](
            data_dir=data_dir,
            patient_ids=train_patients,
            normalizer=NORMALIZERS[config.normalizer](),
            augmentations=augmentations,
            **asdict(config.dataset.args),
        )

        test_dataset = DATASETS[config.dataset.name](
            data_dir=data_dir,
            patient_ids=test_patients,
            normalizer=NORMALIZERS[config.normalizer](),
            augmentations=augmentations,
            **asdict(config.dataset.args),
        )

    if config.batch_size == "sampler":
        train_sampler = SAMPLERS[config.dataset.name](
            window_indices=train_dataset.window_indices
        )

        test_sampler = SAMPLERS[config.dataset.name](
            window_indices=test_dataset.window_indices
        )

        train_dataloader = DataLoader(
            train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=4
        )
        test_dataloader = DataLoader(
            test_dataset, batch_sampler=test_sampler, pin_memory=True, num_workers=4
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    return train_dataloader, test_dataloader, train_patients


def build_model_loss_optimizer_scheduler(
    config: TrainConfig, steps_per_epoch: int, train_patients: list[int] | None = None
):
    if "dann" in config.model.name:
        if train_patients:
            model = MODELS[config.model.name](
                num_subjects=len(train_patients), **config.model.args
            )
        else:
            raise ValueError("number of train subjects must be provided for DANN model")
    else:
        model = MODELS[config.model.name](**config.model.args)

    loss_fn = LOSSES[config.loss.name](**config.loss.args)

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(0.01 * total_steps)

    if config.warmup == "rampup":
        warmup_lr = LinearLR(
            optimizer, start_factor=0.01, end_factor=1, total_iters=warmup_steps
        )
    else:
        warmup_lr = ConstantLR(optimizer, factor=1, total_iters=warmup_steps)

    cosine_lr = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=config.learning_rate * 0.01
    )
    scheduler = SequentialLR(
        optimizer, [warmup_lr, cosine_lr], milestones=[warmup_steps]
    )

    return model, loss_fn, optimizer, scheduler
