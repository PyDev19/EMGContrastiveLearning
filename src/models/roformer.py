import torch
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Parameter,
    Sequential,
)

from src.layers import DropPath, PatchEmbeddings, RotarySelfAttentionBlock
from src.utils.registers import ACTIVATIONS
from src.utils.types import ActivationName


class RotaryTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        proj_drop_prob: float,
        attn_drop_prob: float,
        drop_path_prob: float,
        mlp_drop_prob: float,
        mlp_activation: str = "gelu",
        qkv_bias: bool = False,
    ):
        """Individual transformer block using RoPE self-attention and an MLP with residual connections.

        Args:
            dim (int): embedding dimension of the input and output of the block.
            hidden_dim (int): hidden dimension of the MLP within the block.
            num_heads (int): number of attention heads; must evenly divide dim.
            proj_drop_prob (float): dropout probability for the attention projection.
            attn_drop_prob (float): dropout probability for the attention weights.
            drop_path_prob (float): stochastic depth probability for each block.
            mlp_drop_prob (float): dropout probability for the MLP layers.
            mlp_activation (str, optional): activation function for the MLP. Defaults to "gelu".
            qkv_bias (bool, optional): whether to include a bias term in the QKV projection. Defaults to False.
        """
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = RotarySelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_prob,
            proj_drop=proj_drop_prob,
        )

        self.drop_path1 = DropPath(drop_path_prob)
        self.drop_path2 = DropPath(drop_path_prob)

        self.norm2 = LayerNorm(dim)

        self.mlp = Sequential(
            Linear(dim, hidden_dim),
            ACTIVATIONS[mlp_activation](),
            Dropout(mlp_drop_prob),
            Linear(hidden_dim, dim),
            Dropout(mlp_drop_prob),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for the RotaryTransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim).
            pos_ids (torch.Tensor, optional): Positional IDs for the input sequence. Defaults to None.
            attn_mask (torch.Tensor, optional): Attention mask to apply to the attention weights. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, dim).
        """
        x = x + self.drop_path1(
            self.attn(self.norm1(x), pos_ids=pos_ids, attn_mask=attn_mask)
        )
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class RoFormerConstrastiveModel(Module):
    def __init__(
        self,
        time_steps: int,
        channels: int,
        patch_size: int,
        embed_dim: int,
        hidden_dim: int,
        projection_dim: int,
        projection_hidden_dim: int,
        num_heads: int,
        num_layers: int,
        proj_drop_prob: float,
        attn_drop_prob: float,
        drop_path_prob: float,
        mlp_drop_prob: float,
        qkv_bias: bool = False,
        mlp_activation: ActivationName = "gelu",
        projection_activation: ActivationName = "silu",
    ):
        """ViT-style transformer using RoPE self-attention for contrastive pretraining
        on multi-channel signals. Patches the input, runs it through a stack of
        RotaryTransformerBlocks, mean-pools the token sequence, and projects the
        result through an MLP head for use with a contrastive loss.

        Args:
            time_steps (int): number of timesteps in the input signal (T).
            channels (int): number of input channels (C).
            patch_size (int): patch length along the time axis; must evenly divide time_steps.
            embed_dim (int): transformer hidden/embedding dimension.
            hidden_dim (int): hidden dimension of each block's MLP.
            projection_dim (int): output dimension of the projection head.
            projection_hidden_dim (int): hidden dimension of the projection head.
            num_heads (int): number of attention heads; must evenly divide embed_dim.
            num_layers (int): number of stacked RotaryTransformerBlocks.
            proj_drop_prob (float): dropout probability for the attention projection.
            attn_drop_prob (float): dropout probability for the attention weights.
            drop_path_prob (float): stochastic depth probability for each block.
            mlp_drop_prob (float): dropout probability for the MLP layers.
            qkv_bias (bool, optional): whether QKV projections use bias. Defaults to False.
            mlp_activation (ActivationName, optional): activation used in each block's MLP. Defaults to "gelu".
            projection_activation (ActivationName, optional): activation used in the projection head. Defaults to "silu".
        """
        super().__init__()

        self.num_patches = (time_steps // patch_size) * channels
        self.channel_embed = Parameter(torch.zeros(1, channels, 1, embed_dim))

        self.patch_embedding = PatchEmbeddings(
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.blocks = ModuleList(
            [
                RotaryTransformerBlock(
                    dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_drop_prob=proj_drop_prob,
                    attn_drop_prob=attn_drop_prob,
                    drop_path_prob=drop_path_prob,
                    mlp_drop_prob=mlp_drop_prob,
                    mlp_activation=mlp_activation,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_layer = LayerNorm(embed_dim)

        self.projection_head = Sequential(
            Linear(embed_dim, projection_hidden_dim),
            ACTIVATIONS[projection_activation](),
            Linear(projection_hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the RoFormerConstrastiveModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, projection_dim).
        """
        x = self.patch_embedding(x)
        B, C, P, D = x.shape

        x = x + self.channel_embed[:, :C, :, :]

        x = x.reshape(B, -1, D)

        pos_ids_single = torch.arange(P, device=x.device).repeat(C)
        pos_ids = pos_ids_single.unsqueeze(0).expand(B, -1)

        for blocks in self.blocks:
            x = blocks(x, pos_ids=pos_ids, attn_mask=None)

        x = self.norm_layer(x)

        x = x.mean(dim=1)

        x = self.projection_head(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = RoFormerConstrastiveModel(
        time_steps=1024,
        channels=64,
        embed_dim=256,
        patch_size=64,
        hidden_dim=512,
        num_heads=4,
        num_layers=8,
        proj_drop_prob=0.3,
        attn_drop_prob=0.3,
        drop_path_prob=0.3,
        mlp_drop_prob=0.3,
    )

    summary(
        model,
        input_size=[(1, 64, 1024)],
        depth=4,
        col_names=["input_size", "output_size", "num_params"],
    )
