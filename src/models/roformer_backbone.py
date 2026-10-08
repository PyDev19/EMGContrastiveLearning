import math

import torch
from torch.nn import LayerNorm, Linear, Module, ModuleList, Parameter

from src.layers import DropPath, PatchEmbeddings, RotarySelfAttentionBlock, SwiGLU


class RoFormerBackbone(Module):
    def __init__(
        self,
        time_steps: int,
        channels: int,
        patch_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        proj_drop_prob: float,
        attn_drop_prob: float,
        drop_path_prob: float,
        qkv_bias: bool = False,
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
                len(hidden_dims) determines the number of hidden layers in the MLP.
            num_heads (int): number of attention heads; must evenly divide embed_dim.
            num_layers (int): number of stacked RotaryTransformerBlocks.
            proj_drop_prob (float): dropout probability for the attention projection.
            attn_drop_prob (float): dropout probability for the attention weights.
            drop_path_prob (float): stochastic depth probability for each block.
            qkv_bias (bool, optional): whether QKV projections use bias. Defaults to False.
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
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_layer = LayerNorm(embed_dim)

        torch.nn.init.trunc_normal_(self.channel_embed, std=0.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param: torch.Tensor, layer_id: int):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, block in enumerate(self.blocks):
            rescale(block.attn.projection.weight.data, layer_id + 1)  # type: ignore
            # rescale(block.mlp.layers[-1].weight.data, layer_id + 1)  # type: ignore

    def _init_weights(self, module: Module) -> None:
        """Initialize the weights of the model.

        Args:
            module (torch.nn.Module): The module to initialize.
        """
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the RoFormerConstrastiveModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).

        Returns:
            torch.Tensor: the pooled features.
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

        return x.mean(dim=1)


class RotaryTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        proj_drop_prob: float,
        attn_drop_prob: float,
        drop_path_prob: float,
        qkv_bias: bool = False,
    ):
        """Individual transformer block using RoPE self-attention and an MLP with residual connections.

        Args:
            dim (int): embedding dimension of the input and output of the block.
            hidden_dim (int): hidden dimension of the SwiGLU within the block.
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
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
        )

        self.drop_path1 = DropPath(drop_path_prob)
        self.drop_path2 = DropPath(drop_path_prob)

        self.norm2 = LayerNorm(dim)

        self.swiglu = SwiGLU(
            input_dim=dim,
            hidden_dim=(2 * hidden_dim) // 3,
            output_dim=dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
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
        x = x + self.drop_path2(self.swiglu(self.norm2(x)))
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = RoFormerBackbone(
        time_steps=512,
        channels=64,
        patch_size=64,
        embed_dim=256,
        hidden_dim=1024,
        num_heads=2,
        num_layers=4,
        proj_drop_prob=0.3,
        attn_drop_prob=0.3,
        drop_path_prob=0.3,
    )

    summary(
        model,
        input_size=[(1, 64, 512)],
        depth=4,
        col_names=["input_size", "output_size", "num_params"],
    )
