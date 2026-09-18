import argparse
import pathlib

import numpy as np
import torch
import wandb
from sklearn.metrics import silhouette_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.datasets.physiomio import PhysioMioDataset
from src.loss.masked_reconstruction import MaskedReconstructionLoss
from src.loss.supcon import SupervisedContrastiveLoss
from src.models.roformer_contrastive import RoFormerContrastiveModel
from src.utils.augmentations import Augmentations
from src.utils.registers import NORMALIZERS

DATASETS = {"physiomio": PhysioMioDataset}
MODELS = {"roformer_contrastive": RoFormerContrastiveModel}
LOSSES = {
    "supervised_contrastive": SupervisedContrastiveLoss,
    "masked_reconstruction": MaskedReconstructionLoss,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model based on the provided configuration."
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold number for cross-validation (1-48)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Directory containing the preprocessed patient data files in hdf5 format",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        required=True,
        choices=["zscore", "minmax"],
        help="Which normalizer to use on the data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["physiomio"],
        required=True,
        help="Which dataset to use for training",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["roformer_contrastive"],
        required=True,
        help="Which model to use for training",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["supervised_contrastive", "masked_reconstruction"],
        required=True,
        help="Which loss function to use for training",
    )
    parser.add_argument(
        "--train_task",
        type=str,
        required=True,
        choices=["contrastive", "masked_reconstruction"],
        help="Which training task to perform",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=True,
        help="Weight decay for the optimizer",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (
        args.train_task == "contrastive"
        and args.loss == "supervised_contrastive"
        and args.model == "roformer_contrastive"
    ):
        raise ValueError(
            f"For the 'contrastive' training task, the model must be 'roformer_contrastive' and the loss must be 'supervised_contrastive'. "
            f"Got train_task: {args.train_task}, loss: {args.loss}, model: {args.model}"
        )

    if not (
        args.train_task == "masked_reconstruction"
        and args.loss == "masked_reconstruction"
    ):
        raise ValueError(
            f"For the 'masked_reconstruction' training task, the loss must be 'masked_reconstruction'. "
            f"Got train_task: {args.train_task}, loss: {args.loss}"
        )


def build_datasets(args: argparse.Namespace):
    data = pathlib.Path(args.data)

    normalizer = NORMALIZERS[args.normalizer]
    augmentations = Augmentations()

    patients = [i for i in range(1, 49)]
    test_patient = patients[args.fold - 1]
    train_patients = [p for p in patients if p != test_patient]

    train_dataset = DATASETS[args.dataset](
        data_dir=data,
        patient_ids=train_patients,
        normalizer=normalizer,
        window_opts={"size": 512, "stride": 256},
        augmentations=augmentations,
    )

    test_dataset = DATASETS[args.dataset](
        data_dir=data,
        patient_ids=[test_patient],
        normalizer=normalizer,
        window_opts={"size": 512, "stride": 256},
        augmentations=augmentations,
    )

    return train_dataset, test_dataset


def build_model(args: argparse.Namespace):
    model = MODELS[args.model](
        time_steps=512,
        channels=64,
        patch_size=64,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        hidden_dims=[512],
        projection_hidden_dims=[256],
        projection_dim=128,
        proj_drop_prob=0.1,
        attn_drop_prob=0.1,
        drop_path_prob=0.1,
        mlp_drop_prob=0.1,
        mlp_activation="gelu",
        projection_activation="silu",
    )

    return model


def build_loss(args: argparse.Namespace):
    if args.loss == "supervised_contrastive":
        loss_fn = LOSSES[args.loss](temperature=0.07)
    elif args.loss == "masked_reconstruction":
        loss_fn = LOSSES[args.loss](type="l1")
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    return loss_fn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    validate_args(args)

    torch.manual_seed(args.fold)
    np.random.seed(args.fold)

    train_dataset, test_dataset = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(args)
    model.to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = build_loss(args)

    num_epochs = args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    run = wandb.init(
        entity="atharvamishra-university-of-texas-at-dallas",
        project="ProjectBlueprint",
        config={
            "epochs": num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "normalizer": args.normalizer,
            "optimizer": "AdamW",
            "model": args.model,
            "loss": args.loss,
            "dataset": args.dataset,
        },
    )

    run.watch(model, loss_fn, log="all", log_freq=10)

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        train_samples = 0
        for batch in train_loader:
            optimizer.zero_grad()

            emg, emg_aug, labels = batch
            emg = emg.to(device)
            emg_aug = emg_aug.to(device)
            labels = labels.to(device)

            _, z = model(emg)
            _, z_aug = model(emg_aug)

            loss = loss_fn(z, z_aug, labels)
            loss.backward()
            optimizer.step()

            batch_size = emg.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss / train_samples
        scheduler.step()

        run.log({"train_loss": train_loss}, step=epoch)

        model.eval()

        test_loss = 0
        test_samples = 0
        all_h = []
        all_labels = []
        for batch in test_loader:
            with torch.no_grad():
                emg, emg_aug, labels = batch
                emg = emg.to(device)
                emg_aug = emg_aug.to(device)
                labels = labels.to(device)

                h, z = model(emg)
                _, z_aug = model(emg_aug)

                loss = loss_fn(z, z_aug, labels)

            batch_size = emg.size(0)
            test_loss += loss.item() * batch_size
            test_samples += batch_size

            all_h.append(h.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        test_loss = test_loss / test_samples
        all_h = np.concatenate(all_h, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        test_silhouette = silhouette_score(all_h, all_labels)

        run.log(
            {"test_loss": test_loss, "test_silhouette": test_silhouette}, step=epoch
        )

    run.finish()


if __name__ == "__main__":
    main()
