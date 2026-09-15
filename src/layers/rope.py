import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Module


class RotaryPositionalEmbeddings(Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: int = 10_000):
        """Implementation of RoPe as described in https://arxiv.org/pdf/2104.09864.
        In this implementation embeddings are cached up to `max_seq_len` during initialization.

        Args:
            dim (int): Dimension of embeddings per token.
            max_seq_len (int, optional): maximum length of sequences expected. Defaults to 1024.
            base (int, optional): base value for geometric progression in cache, same value used in the paper. Defaults to 10_000.
        """
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
        """Runs the forward pass for RoPe on Q and K vectors. If input positions are provided
        they are fetched from the cache, if not [0, seq_len-1] are fetched from the cache

        Args:
            x (torch.Tensor): float tensor, either Q or K, of shape [B, H, S, D] where
                                H is heads, S is seq_len, and D is dimension per head
            input_pos (torch.Tensor | None, optional): input position vector per sequence in batch of shape [B, S]. Defaults to None.

        Returns:
            torch.Tensor: output tensor encoded with rotary positions of shape [B, H, S, D]
        """

        seq_len = x.size(2)  # extract seq_len

        if input_pos is None:
            rope_cache = self.cache[:seq_len]  # (S, head_dim//2, 2)

            # add batch and head dims for broadcasting against (B, H, S, head_dim//2, 2)
            # shape (1, 1, S, head_dim//2, 2)
            rope_cache = rope_cache.unsqueeze(0).unsqueeze(0)
        else:
            input_pos = input_pos.to(torch.long)  # (B, S)
            rope_cache = self.cache[input_pos]  # (B, S, head_dim//2, 2)

            # add head dim only, batch dim already present from advanced indexing
            rope_cache = rope_cache.unsqueeze(1)  # (B, 1, S, head_dim//2, 2)

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (B, H, S, head_dim//2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # undoing the earlier pair-split
        x_out = x_out.flatten(3)  # (B, H, S, D)
        return x_out.type_as(x)


class RotarySelfAttentionBlock(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop_prob: float,
        proj_drop_prob: float,
        qkv_bias: bool = False,
    ):
        """Implementation of Multi-Head Attention with RoPe.

        Args:
            dim (int): embedding dimension of each token.
            num_heads (int): number of attention heads.
            attn_drop_prob (float): dropout probability during scaled dot product attention.
            proj_drop_prob (float): dropout probability during project of the output tensor.
            qkv_bias (bool, optional): whether bias should be used in QKV linear layer. Defaults to False.
        """
        super().__init__()
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop_prob
        self.head_dim = dim // num_heads

        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,
            max_seq_len=1024,
            base=10_000,
        )

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)

        self.projection = Linear(dim, dim)
        self.projection_dropout = Dropout(proj_drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Runs the forward pass of MHA but with RoPe encodings on Q and K.

        Args:
            x (torch.Tensor): float sequence tensor after being patched, shape (B, S, D)
            pos_ids (torch.Tensor, optional): int position vector per sequence in batch of shape [B, S]. Defaults to None.
            attn_mask (torch.Tensor, optional): boolean attention mask of shape (B, 1, S, S).
            True/nonzero positions are attended to, following F.scaled_dot_product_attention's convention. Defaults to None.

        Returns:
            torch.Tensor: output tensor after QKV self-attention calculation, shape (B, S, D)
        """
        B, S, D = x.shape  # batch, seq_len, dim
        qkv = (
            self.qkv(x)
            .reshape(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)  # move QKV to front for unbind
        )  # (3, B, n_heads, S, head_dim)
        q, k, v = qkv.unbind(0)  # each: (B, n_heads, S, head_dim)

        # unchanged shape, vectors are just moved by theta
        q = self.rope(q, input_pos=pos_ids)
        k = self.rope(k, input_pos=pos_ids)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
            is_causal=False,
            enable_gqa=False,
        )  # (B, n_heads, S, head_dim)

        x = x.transpose(2, 1).reshape(B, S, D)
        x = self.projection(x)
        x = self.projection_dropout(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    rope_attn = RotarySelfAttentionBlock(dim=256, num_heads=4)

    summary(
        rope_attn,
        input_size=(1, 1024, 256),
        depth=4,
        col_names=["input_size", "output_size", "num_params"],
    )
