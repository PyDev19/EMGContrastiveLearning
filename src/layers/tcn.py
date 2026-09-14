import torch
import torch.nn.functional as F
from torch.nn import Conv1d, Dropout, Identity, Module, Sequential

from src.utils import ACTIVATIONS, NORM_LAYERS


class TCNBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str = "batch",
        groups: int = 8,
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.activation = ACTIVATIONS[activation]()
        self.dropout = Dropout(dropout)

        self.conv1 = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.norm1 = (
            NORM_LAYERS[norm](groups, out_channels)
            if norm == "group"
            else NORM_LAYERS[norm](out_channels)
        )

        self.conv2 = Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.norm2 = (
            NORM_LAYERS[norm](groups, out_channels)
            if norm == "group"
            else NORM_LAYERS[norm](out_channels)
        )

        self.residual = (
            Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = F.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.norm2(out)

        res = self.residual(x)

        return self.activation(out + res)


class TCNLayer(Module):
    def __init__(
        self,
        channels: list[int],
        dilations: list[int],
        kernel_size: int,
        dropout: float = 0.2,
        activation: str = "relu",
        norm: str = "batch",
        groups: int = 8,
    ):
        super().__init__()

        assert len(dilations) == len(channels) - 1, (
            f"Expected {len(channels) - 1} dilations for {len(channels)} channels, "
            f"got {len(dilations)}"
        )

        self.layers = Sequential()

        for i in range(len(channels) - 1):
            self.layers.append(
                TCNBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    groups=groups,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
