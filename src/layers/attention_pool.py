import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter


class AttentionPool(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        attn_drop_prob: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop_prob
        self.query = Parameter(torch.zeros(1, 1, dim))

        self.q_proj = Linear(dim, dim, bias=False)
        self.kv_proj = Linear(dim, 2 * dim, bias=False)

        self.projection = Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        q = self.q_proj(self.query).expand(B, -1, -1)  # (B, 1, D)
        k, v = self.kv_proj(x).chunk(2, dim=-1)  # each (B, N, D)

        q = q.reshape(B, 1, H, D // H).transpose(1, 2)  # (B, H, 1, D / H)
        k = k.reshape(B, N, H, D // H).transpose(1, 2)  # (B, H, N, D / H)
        v = v.reshape(B, N, H, D // H).transpose(1, 2)  # (B, H, 1, D / H)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
            is_causal=False,
            enable_gqa=False,
        )  # (B, H, 1, D / H)

        return self.projection(out.transpose(1, 2).reshape(B, D))
