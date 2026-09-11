from dataclasses import dataclass

import torch
from torch.nn import Conv2d, Module
from torch.nn.init import xavier_uniform_


@dataclass(eq=False)
class PatchEmbeddings(Module):        
    time_steps: int
    patch_size: int
    channels: int
    embed_dim: int

    def __post_init__(self):
        super().__init__()

        self.num_patches = (self.time_steps // self.patch_size) * self.channels
        self.projections = Conv2d(
            1,
            self.embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
        )
        
        # initialize projections like Linear (instead of Conv2d default),
        # since kernel_size == stride means each patch is projected independently —
        # functionally a linear layer, not a true (overlapping) convolution
        w = self.projections.weight.data
        xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projections(x.unsqueeze(1))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        return x

if __name__ == "__main__":
    embedding = PatchEmbeddings(time_steps=1024, patch_size=8, channels=64, embed_dim=256)

    sample = torch.randn(1, 64, 1024)
    out = embedding(sample)

    print(sample.shape)
    print(out.shape)
