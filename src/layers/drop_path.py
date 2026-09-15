import torch
from torch.nn import Module


class DropPath(Module):
    def __init__(self, drop_prob: float):
        """Implementaion of stochastic depth layer from https://arxiv.org/pdf/1603.09382

        Args:
            drop_prob (float): The probrabilty that an item is dropped

        Raises:
            ValueError: If the probability isn't between 0 and 1
        """
        super().__init__()

        if drop_prob > 1 or drop_prob < 0:
            raise ValueError("Drop path probability should be betweeen 0 and 1")

        self.keep_probability = 1 - drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.keep_probability == 1 or not self.training:
            return x

        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(self.keep_probability)
        mask.div_(self.keep_probability)

        return x * mask
