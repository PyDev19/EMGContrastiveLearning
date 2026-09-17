import pathlib

import numpy as np
import torch
import wandb
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    run.watch(model, loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        test_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            emg, emg_aug, labels = batch
            emg = emg.to(device)
            emg_aug = emg_aug.to(device)
            labels = labels.to(device)

            z = model(emg)
            z_aug = model(emg_aug)

            loss = loss_fn(z, z_aug, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                emg, emg_aug, labels = batch
                emg = emg.to(device)
                emg_aug = emg_aug.to(device)
                labels = labels.to(device)

                z = model(emg)
                z_aug = model(emg_aug)

                loss = loss_fn(z, z_aug, labels)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        run.log({"epoch": epoch + 1, "train_loss": train_loss, "test_loss": test_loss})

    run.finish()


if __name__ == "__main__":
    main()
