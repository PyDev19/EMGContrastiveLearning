import torch
from torch.nn import Linear, Module


class SwiGLU(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """SwiGLU feedforward block (Shazeer, https://arxiv.org/abs/2002.05202),
        as used in PaLM/LLaMA-style transformers. A gated variant of a standard
        MLP: uses three linear layers (gate, value, down-projection) instead of
        two, with SiLU(gate) ⊙ value as the gating mechanism in place of a plain
        pointwise activation.

        Args:
            input_dim (int): input/output dimension (matches the residual stream).
            hidden_dim (int): internal gating dimension. Note this has a different
                parameter-count relationship to input_dim than a plain MLP's
                hidden_dims — to match a plain MLP's parameter count, use roughly
                (2/3) of the hidden_dim you'd otherwise choose.
            output_dim (int): output dimension.
        """
        super().__init__()

        self.gate_proj = Linear(input_dim, hidden_dim, bias=False)
        self.value_proj = Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.value_proj(x)
        return self.down_proj(gate * value)
