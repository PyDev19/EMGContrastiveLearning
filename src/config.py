from dataclasses import dataclass
from typing import Literal, cast

from omegaconf import OmegaConf

from src.utils.types import ActivationName, NormalizerName, WindowOpts


@dataclass
class DatasetArgs:
    window_opts: WindowOpts
    rms_opts: WindowOpts | None = None


@dataclass
class DatasetConfig:
    name: Literal["physiomio"]
    args: DatasetArgs


@dataclass
class RoFormerContrastiveModelArgs:
    time_steps: int
    channels: int
    patch_size: int
    embed_dim: int
    hidden_dims: list[int]
    projection_dim: int
    projection_hidden_dims: list[int]
    num_heads: int
    num_layers: int
    proj_drop_prob: float
    attn_drop_prob: float
    drop_path_prob: float
    mlp_drop_prob: float
    qkv_bias: bool
    mlp_activation: ActivationName
    projection_activation: ActivationName


@dataclass
class ModelConfig:
    name: Literal["roformer_contrastive"]
    args: RoFormerContrastiveModelArgs


@dataclass
class SupervisedContrastiveLossArgs:
    temperature: float


@dataclass
class LossConfig:
    name: Literal["supervised_contrastive"]
    args: SupervisedContrastiveLossArgs


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
class ContrastiveTrainingConfig:
    task: Literal["contrastive"]
    batch_size: int
    num_epochs: int
    normalizer: NormalizerName
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerArgs
    wandb: WandbConfig


def load_training_config(config_path: str) -> ContrastiveTrainingConfig:
    """Load and validate a contrastive-training config from YAML. Every field
    is required — no defaults are baked into the schema, so every experiment
    YAML is fully self-documenting and sweep-friendly.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        ContrastiveTrainingConfig: the validated, fully-typed training config.

    Raises:
        ValueError: if `task` is missing or not "contrastive".
    """
    raw = OmegaConf.load(config_path)

    task = OmegaConf.select(raw, "task")
    if task != "contrastive":
        raise ValueError(
            f"Unknown or missing 'task' in config: {task!r}. Expected 'contrastive'."
        )

    merged = OmegaConf.merge(OmegaConf.structured(ContrastiveTrainingConfig), raw)
    return cast(ContrastiveTrainingConfig, OmegaConf.to_object(merged))


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
