from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf


@dataclass
class DatasetConfig:
    window_size: int = 512
    stride: int = 256
    
    jitter_sigma: float = 0.01
    scale_sigma: float = 0.1
    mask_prob: float = 0.1
    
    freq_perturb_ratio: float = 0.1
    freq_alpha: float = 0.1


@dataclass
class LossConfig:
    temperature: float = 0.2
    margin: int = 1
    contrastive_weight: float = 0.2


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.99)


@dataclass
class BaseTrainConfig:
    epochs: int = 100
    train_batch_size: int = 64
    val_batch_size: int = 128
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class TransformerCLSConfig:
    name: str = "transformer_cls"
    time_dim: int = 512
    hidden_dim: int = 256
    latent_dim: int = 128
    num_layers: int = 2
    nheads: int = 2
    dropout: float = 0.1

@dataclass
class TransformerCLSTrainConfig(BaseTrainConfig):
    model: TransformerCLSConfig = field(default_factory=TransformerCLSConfig)


CONFIG_REGISTRY = {
    "transformer_cls": TransformerCLSTrainConfig,
}


def load_config(config_path: str) -> DictConfig:
    raw: DictConfig = OmegaConf.load(config_path)

    model_name = raw.get("model", {}).get("name")
    config_cls = CONFIG_REGISTRY.get(model_name)
    if config_cls is None:
        raise ValueError(
            f"Unknown model name {model_name!r} in config. "
            f"Expected one of: {list(CONFIG_REGISTRY)}"
        )

    return OmegaConf.merge(OmegaConf.structured(config_cls()), raw)


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Save default configurations for TCN and Transformer TFC models"
    )
    parser.add_argument("--model", choices=list(CONFIG_REGISTRY), default="transformer_cls")
    parser.add_argument("--output", default="config.json")
    args = parser.parse_args()

    config = OmegaConf.structured(CONFIG_REGISTRY[args.model]())
    with open(args.output, "w") as f:
        json.dump(OmegaConf.to_container(config), f, indent=4)

    config = load_config(args.output)
    print(config)
