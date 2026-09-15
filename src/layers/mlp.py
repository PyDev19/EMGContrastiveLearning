import torch
from torch.nn import Dropout, Linear, Module, ModuleList

from src.utils.registers import ACTIVATIONS
from src.utils.types import ActivationName


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: ActivationName | None = None,
        dropout: float | None = None,
    ):
        """Initialize a multi-layer perceptron (MLP) model.

        Args:
            input_dim (int): The dimension of the input features.
            hidden_dims (list[int]): A list containing the dimensions of the hidden layers. len(hidden_dims) determines the number of hidden layers.
            output_dim (int): The dimension of the output layer.
            activation (ActivationName | None, optional): The activation function to apply after each hidden layer. Defaults to None.
            dropout (float | None, optional): The dropout probability to apply after each hidden layer. Defaults to None.
        """
        super().__init__()
        self.layers = ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(Linear(prev_dim, hidden_dim))

            if activation is not None:
                self.layers.append(ACTIVATIONS[activation]())

            if dropout is not None:
                self.layers.append(Dropout(dropout))

            prev_dim = hidden_dim

        self.layers.append(Linear(prev_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim) after passing through the MLP model.
        """
        for layer in self.layers:
            x = layer(x)

        return x
