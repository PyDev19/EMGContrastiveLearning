import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Module


class RotaryPositionalEmbeddings(Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10_000):
        super().__init__()

        theta = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
        )
        self.register_buffer("theta", theta, persistent=False)

        seq_idx = torch.arange(0, max_seq_len, dtype=torch.float32)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.size(1)

        if input_pos is None:
            rope_cache = self.cache[:seq_len]
        else:
            input_pos = input_pos.to(torch.long)
            rope_cache = self.cache[input_pos]

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class RotarySelfAttentionBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        head_dim = dim // num_heads

        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim,
            max_seq_len=1024,
            base=10_000,
        )

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attention_dropout = Dropout(attn_drop)

        self.projection = Linear(dim, dim)
        self.projection_dropout = Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = self.rope(q)
        k = self.rope(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=False,
        )

        x = x.transpose(2, 1).reshape(B, N, C)
        x = self.projection(x)
        x = self.projection_dropout(x)

        return x
