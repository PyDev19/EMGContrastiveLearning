import torch
from torch.nn import (
    BatchNorm1d,
    Linear,
    Module,
    Parameter,
    ReLU,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
    Dropout,
)


class CLSTransformerTFC(Module):
    def __init__(
        self,
        time_dim: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        nheads: int = 2,
        dropout: float = 0.1,
    ):
        self.time_cls_token = Parameter(torch.zeros(1, 1, time_dim))
        self.time_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=time_dim,
                nhead=nheads,
                dim_feedforward=time_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.time_projector = Sequential(
            Linear(time_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, latent_dim),
        )

        self.frequency_cls_token = Parameter(torch.zeros(1, 1, time_dim))
        self.frequency_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=time_dim,
                nhead=nheads,
                dim_feedforward=time_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.frequency_projector = Sequential(
            Linear(time_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, latent_dim),
        )

    def forward(self, time_input: torch.Tensor, frequency_input: torch.Tensor):
        batch_size = time_input.size(0)

        time_cls_tokens = self.time_cls_token.expand(batch_size, -1, -1)
        time_input = torch.cat([time_cls_tokens, time_input], dim=1)
        ht = self.time_encoder(time_input)
        ht = ht[:, 0, :]
        zt = self.time_projector(ht)

        frequency_cls_tokens = self.frequency_cls_token.expand(batch_size, -1, -1)
        frequency_input = torch.cat([frequency_cls_tokens, frequency_input], dim=1)
        hf = self.frequency_encoder(frequency_input)
        hf = hf[:, 0, :]
        zf = self.frequency_projector(hf)

        return ht, zt, hf, zf
