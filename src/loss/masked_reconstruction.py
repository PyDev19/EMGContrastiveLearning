from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn import Module


class MaskedReconstructionLoss(Module):
    def __init__(self, loss_type: Literal["l1", "l2", "smooth_l1"]):
        """Masked Reconstruction Loss for comparing reconstructed and original signals, separately for masked and unmasked elements.

        Args:
            loss_type (Literal["l1", "l2", "smooth_l1"]): Type of loss to compute. Options are:
                - "l1": Mean Absolute Error (L1 loss)
                - "l2": Mean Squared Error (L2 loss)
                - "smooth_l1": Smooth L1 (Huber) loss
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the masked reconstruction loss between the reconstructed and original signals, separately for masked and unmasked elements.

        Args:
            reconstructed (torch.Tensor): The reconstructed signal tensor of shape (batch_size, seq_length, channels).
            original (torch.Tensor): The original signal tensor of shape (batch_size, seq_length, channels).
            mask (torch.BoolTensor): Boolean mask indicating which elements are masked.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The loss for masked and unmasked elements, respectively.
        """
        if self.loss_type == "l1":
            masked_loss = F.l1_loss(
                reconstructed[mask], original[mask], reduction="mean"
            )
            unmasked_loss = F.l1_loss(
                reconstructed[~mask], original[~mask], reduction="mean"
            )
        elif self.loss_type == "l2":
            masked_loss = F.mse_loss(
                reconstructed[mask], original[mask], reduction="mean"
            )
            unmasked_loss = F.mse_loss(
                reconstructed[~mask], original[~mask], reduction="mean"
            )
        elif self.loss_type == "smooth_l1":
            masked_loss = F.smooth_l1_loss(
                reconstructed[mask], original[mask], reduction="mean"
            )
            unmasked_loss = F.smooth_l1_loss(
                reconstructed[~mask], original[~mask], reduction="mean"
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return masked_loss, unmasked_loss
