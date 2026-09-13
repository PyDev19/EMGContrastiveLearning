import torch
from torch.nn import Conv2d, Module
from torch.nn.init import xavier_uniform_


class PatchEmbeddings(Module):
    def __init__(self, patch_size: int, embed_dim: int):
        """ViT-style patch embedding for multi-channel 1D signals, implemented as a Conv2D with non-overlapping kernels.

        The input is treated as a single-channel image of shape (B, 1, C, T)
        and convolved with kernel == stride == (1, patch_size). The convolution only happens
        at the channel level to be channel independent and non-overlapping kernels effectively
        makes this a classic linear-style projection so we use Xavier init for weights.

        The tokens outputed is equal to (T // patch_size) * C, but the final output isn't flattened.
        The channels and patches dimension is kept for further calculations.

        Args:
            patch_size (int): size of patches which will be used for kernel and stride
            embed_dim (int): embedding vector dimension per patch
        """
        super().__init__()

        self.projections = Conv2d(
            1,
            embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
        )

        # initialize projections like Linear (instead of Conv2d default),
        # since kernel_size == stride means each patch is projected independently —
        # functionally a linear layer, not a true (overlapping) convolution
        w = self.projections.weight.data
        xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input signal, shape (B, C, T).

        Returns:
            torch.Tensor: patch embeddings, shape (B, C, P, D), where
                P = T // patch_size and D = embed_dim. Channel and patch
                dimensions are kept separate rather than flattened.
        """

        x = self.projections(x.unsqueeze(1))
        x = x.permute(0, 2, 3, 1)

        return x


if __name__ == "__main__":
    embedding = PatchEmbeddings(patch_size=8, embed_dim=256)

    sample = torch.randn(1, 64, 1024)
    out = embedding(sample)

    print(sample.shape)
    print(out.shape)
