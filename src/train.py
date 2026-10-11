import argparse
import pathlib

import numpy as np
import torch
import wandb

from src.config import (
    build_dataloaders,
    build_loss,
    build_model_optimizer_scheduler,
    load_training_config,
)
from src.contrastive_trainer import TimeContrastiveTrainer


def main():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=str, required=True)
    args, _unknown = parser.parse_known_args()

    base = load_training_config(args.config)
    run = wandb.init(entity=base.wandb.entity, project=base.wandb.project)

    config = load_training_config(args.config, overrides=dict(run.config))
    run.name = f"{config.task}_{config.model.name}_{run.id}"

    train_dataloader, test_dataloader = build_dataloaders(config, args.data_dir)
    model, optimizer, scheduler = build_model_optimizer_scheduler(config)
    loss_fn = build_loss(config)

    match config.task:
        case "time_contrastive":
            trainer = TimeContrastiveTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                config=config,
                run=run,
            )
        case _:
            raise NotImplementedError(f"No trainer for {config.task}")

    trainer.train()


if __name__ == "__main__":
    main()
