from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Module


@dataclass(eq=False)
class RotaryPositionalEmbeddings(Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    dim: int
    max_seq_len: int = 4096
    base: int = 10_000

    def __post_init__(self):
        super().__init__()

        dim = int(self.dim)
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
        )
        self.register_buffer("theta", theta, persistent=False)

        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(0, self.max_seq_len, dtype=torch.float32)  # type: ignore

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache: torch.Tensor = torch.stack(
            [torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1
        )
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        if input_pos is None:
            rope_cache = self.cache[:seq_len]  # type: ignore
        else:
            input_pos = input_pos.to(torch.long)
            rope_cache = self.cache[input_pos]  # type: ignore

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


@dataclass(eq=False)
class RotarySelfAttentionBlock(Module):
    """
    A self-attention block that incorporates rotary positional embeddings (RoPE) for enhanced positional awareness.

    This module implements multi-head self-attention with rotary positional embeddings applied to query and key tensors,
    followed by scaled dot-product attention. It is designed for transformer-based architectures, particularly in vision
    or sequence modeling tasks where positional information is crucial.

    Attributes:
        dim (int): The dimensionality of the input and output features.
        num_heads (int): The number of attention heads.
        rope (RotaryPositionalEmbeddings): The rotary positional embedding module.
        scale (float): The scaling factor for attention logits.
        qkv (Linear): Linear layer for projecting input to query, key, and value.
        attn_drop_fn (Dropout): Dropout layer for attention weights.
        proj (Linear): Linear layer for projecting attention output.
        proj_drop (Dropout): Dropout layer for projection output.
    """

    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    def __post_init__(self):
        super().__init__()
        head_dim = self.dim // self.num_heads
        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim,
            max_seq_len=1024,
            base=10_000,
        )
        self.qkv = Linear(self.dim, self.dim * 3, bias=self.qkv_bias)
        self.attn_drop_fn = Dropout(self.attn_drop)
        self.proj = Linear(self.dim, self.dim)
        self.p_drop = Dropout(self.proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for rotary self-attention block."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (K, B, H, N, D)
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D)

        q = self.rope(q)
        k = self.rope(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
            enable_gqa=False,
        )

        x = x.transpose(2, 1).reshape(B, N, C)
        x = self.proj(x)
        x = self.p_drop(x)
        return x
