import torch
from torch.nn import (
    Linear,
    Module,
    Parameter,
)

from src.layers import (
    MLP,
    AttentionPool,
    GradientReverseLayer,
    PatchEmbeddings,
)
from src.models.roformer_backbone import RoFormerBackbone
from src.utils.registers import MODELS, register
from src.utils.types import ActivationName, NormLayerName


@register(MODELS, "roformer_time_contrastive")
class RoFormerTimeContrastiveModel(Module):
    def __init__(
        self,
        time_steps: int,
        channels: int,
        patch_size: int,
        embed_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_heads: int,
        num_layers: int,
        proj_drop_prob: float,
        attn_drop_prob: float,
        drop_path_prob: float,
        head_drop_prob: float,
        pool_drop_prob: float,
        projection_activation: ActivationName = "silu",
        norm: NormLayerName = "layer",
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
            hidden_dim (int): hidden dimension of the MLP in each block.
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

        self.roformer_backbone = RoFormerBackbone(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            proj_drop_prob=proj_drop_prob,
            attn_drop_prob=attn_drop_prob,
            drop_path_prob=drop_path_prob,
            norm=norm,
            qkv_bias=qkv_bias,
        )

        self.attention_pool = AttentionPool(
            dim=embed_dim, attn_drop_prob=pool_drop_prob
        )

        self.projection_head = MLP(
            input_dim=embed_dim,
            hidden_dims=[embed_dim * 2],
            output_dim=projection_dim,
            activation=projection_activation,
            dropout=head_drop_prob,
        )

        self.projection_head.apply(self._init_weights)

        torch.nn.init.trunc_normal_(self.channel_embed, std=0.02)

    def _init_weights(self, module: Module) -> None:
        """Initialize the weights of the model.

        Args:
            module (torch.nn.Module): The module to initialize.
        """
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, return_projected: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for the RoFormerConstrastiveModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
            return_projected (bool): Whether to return the projected features.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing the pooled features and the projected features.
        """
        x = self.patch_embedding(x)
        _, C, _, _ = x.shape
        x = x + self.channel_embed[:, :C, :, :]

        pooled = self.attention_pool(self.roformer_backbone(x))
        projection = self.projection_head(pooled) if return_projected else None

        return pooled, projection


@register(MODELS, "roformer_time_dann_contrastive")
class RoFormerTimeDomainAdversialContrastiveModel(RoFormerTimeContrastiveModel):
    def __init__(self, *, num_subjects: int, **kwargs):
        """Same as RoFormerTimeContrastiveModel, plus a gradient-reversal
        subject classifier on the pooled features.

        Args:
            num_subjects (int): number of subject classes for the adversarial head.
            **kwargs: forwarded to RoFormerTimeContrastiveModel.
        """
        super().__init__(**kwargs)

        self.gradient_reverse = GradientReverseLayer()
        self.subject_adversial_head = MLP(
            input_dim=kwargs["embed_dim"],
            hidden_dims=[],
            output_dim=num_subjects,
        )
        self.subject_adversial_head.apply(self._init_weights)

    def forward(
        self, x: torch.Tensor, return_projected: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        pooled, projection = super().forward(x, return_projected)

        domain_logits = None
        if return_projected:
            domain_logits = self.subject_adversial_head(self.gradient_reverse(pooled))

        return pooled, projection, domain_logits
