import math

import torch
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Parameter,
)

from src.layers import (
    MLP,
    GradientReverseLayer,
    PatchEmbeddings,
)
from src.layers.rope import RotaryTransformerBlock
from src.utils.types import ActivationName


class RoFormerContrastiveModel(Module):
    def __init__(
        self,
        time_steps: int,
        channels: int,
        patch_size: int,
        embed_dim: int,
        hidden_dims: list[int],
        projection_dim: int,
        projection_hidden_dims: list[int],
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
            hidden_dims (list[int]): hidden dimensions of each block's MLP.
                len(hidden_dims) determines the number of hidden layers in the MLP.
            projection_dim (int): output dimension of the projection head.
            projection_hidden_dims (list[int]): hidden dimensions of the projection head.
                len(projection_hidden_dims) determines the number of hidden layers in the projection head.
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
                    hidden_dims=hidden_dims,
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

        self.projection_head = MLP(
            input_dim=embed_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=projection_dim,
            activation=projection_activation,
            dropout=mlp_drop_prob,
        )

        self.subject_discriminator = GradientReverseLayer()

        self.subject_head = MLP(
            input_dim=embed_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=projection_dim,
            activation=projection_activation,
            dropout=mlp_drop_prob,
        )

        torch.nn.init.trunc_normal_(self.channel_embed, std=0.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param: torch.Tensor, layer_id: int):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, block in enumerate(self.blocks):
            rescale(block.attn.projection.weight.data, layer_id + 1)  # type: ignore
            rescale(block.mlp.layers[-1].weight.data, layer_id + 1)  # type: ignore

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

    def forward(
        self, x: torch.Tensor, return_projected: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward pass for the RoFormerConstrastiveModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            return_projected (bool): Whether to return the projected features.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: A tuple containing the pooled features, the projected features, and the subject projection.
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

        pooled = x.mean(dim=1)

        gresture_projection = None
        subject_projection = None

        if return_projected:
            gresture_projection = self.projection_head(pooled)
            subject_projection = self.subject_discriminator(pooled)
            subject_projection = self.subject_head(subject_projection)

        return pooled, gresture_projection, subject_projection


if __name__ == "__main__":
    from torchinfo import summary

    model = RoFormerContrastiveModel(
        time_steps=1024,
        channels=64,
        embed_dim=256,
        patch_size=64,
        hidden_dims=[512],
        projection_hidden_dims=[256],
        projection_dim=128,
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
