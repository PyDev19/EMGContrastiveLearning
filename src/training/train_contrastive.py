import pathlib

import numpy as np
import torch
import wandb
from sklearn.metrics import silhouette_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.datasets.physiomio import PhysioMioDataset
from src.loss.supcon import SupervisedContrastiveLoss
from src.models.roformer_contrastive import RoFormerConstrastiveModel
from src.utils.registers import NORMALIZERS


def main():
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1024
    DATA_DIR = pathlib.Path("data/physiomio")
    NORMALIZER = "zscore"
    TEMPERATURE = 0.07

    run = wandb.init(
        entity="atharvamishra-university-of-texas-at-dallas",
        project="ProjectBlueprint",
        config={
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "normalizer": NORMALIZER,
            "optimizer": "AdamW",
            "model": "RoFormerContrastiveModel",
            "temperature": TEMPERATURE,
        },
    )

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patients = np.array([i for i in range(1, 49)])
    np.random.shuffle(patients)

    test_patient = np.random.choice(patients, 1)
    patients = patients[patients != test_patient]

    normalizer = NORMALIZERS[NORMALIZER]()

    train_dataset = PhysioMioDataset(
        data_dir=DATA_DIR,
        patient_ids=patients.tolist(),
        window_opts={"size": 512, "stride": 256},
        normalizer=normalizer,
    )

    test_dataset = PhysioMioDataset(
        data_dir=DATA_DIR,
        patient_ids=test_patient.tolist(),
        window_opts={"size": 512, "stride": 256},
        normalizer=normalizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    model = RoFormerConstrastiveModel(
        time_steps=512,
        channels=64,
        patch_size=64,
        embed_dim=256,
        hidden_dims=[512],
        projection_dim=128,
        projection_hidden_dims=[512],
        num_heads=4,
        num_layers=8,
        proj_drop_prob=0.3,
        attn_drop_prob=0.3,
        drop_path_prob=0.3,
        mlp_drop_prob=0.3,
    ).to(device)

    optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE)
    loss_fn = SupervisedContrastiveLoss(temperature=TEMPERATURE)

    run.watch(model, loss_fn, log="all", log_freq=10)

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            emg, emg_aug, labels = batch
            emg = emg.to(device)
            emg_aug = emg_aug.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            _, z = model(emg)
            _, z_aug = model(emg_aug)

            loss = loss_fn(z, z_aug, labels)

            loss.backward()
            optimizer.step()

            batch_size = emg.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss = train_loss / train_samples

        model.eval()

        test_loss = 0.0
        test_samples = 0
        all_test_embeddings = []
        all_test_labels = []

        for batch in test_loader:
            emg, emg_aug, labels = batch
            emg = emg.to(device)
            emg_aug = emg_aug.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                h, z = model(emg)
                h_aug, z_aug = model(emg_aug)

                loss = loss_fn(z, z_aug, labels)

                all_test_embeddings.append(h.cpu().numpy())
                all_test_embeddings.append(h_aug.cpu().numpy())
                all_test_labels.append(labels.cpu().numpy())
                all_test_labels.append(labels.cpu().numpy())

            batch_size = emg.size(0)
            test_loss += loss.item() * batch_size
            test_samples += batch_size

        test_loss = test_loss / test_samples
        test_silhouette_score = silhouette_score(
            np.concatenate(all_test_embeddings, axis=0),
            np.concatenate(all_test_labels, axis=0),
        )
        run.log(
            {
                "test_loss": test_loss,
                "test_silhouette_score": test_silhouette_score,
                "train_loss": train_loss,
            },
            step=epoch,
        )

    run.finish()


if __name__ == "__main__":
    main()
