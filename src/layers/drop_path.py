import torch
from torch.nn import Module


class DropPath(Module):
    def __init__(self, probability: float):
        if probability > 1 or probability < 0:
            raise ValueError("Drop path probability should be betweeen 0 and 1")

        self.keep_probability = 1 - probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x

        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(self.keep_probability)
        mask.div_(self.keep_probability)

        return x * mask
