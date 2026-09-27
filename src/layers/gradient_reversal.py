from typing import Any

import torch
from torch.autograd import Function
from torch.nn import Module


class GradientReverseFunction(Function):
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: float | None = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_output = grad_outputs[0]
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
