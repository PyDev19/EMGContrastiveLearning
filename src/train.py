import argparse
import datetime
import pathlib
from dataclasses import asdict

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_training_config(str(args.config))

    torch.manual_seed(42)
    np.random.seed(42)

    train_dataloader, test_dataloader = build_dataloaders(config, args.data_dir)
    model, optimizer, scheduler = build_model_optimizer_scheduler(config)
    loss_fn = build_loss(config)

    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        name=f"{config.task}_{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d_%H%M%S')}",
        config={**asdict(config)},
    )
    run.watch(model, loss_fn, log="all", log_freq=config.wandb.log_freq)

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
