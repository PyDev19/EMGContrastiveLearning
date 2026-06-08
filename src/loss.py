import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy


class TFCLoss(Module):
    def __init__(self, temperature=0.2, margin=1, contrastive_weight=0.2):
        """Initializes the TFCLoss module with the specified hyperparameters.

        Args:
            temperature (float, optional): The temperature parameter for the contrastive loss to scale the similarities. Defaults to 0.2.
            margin (int, optional): The margin for the consistency loss to enforce the separation between positive and negative pairs. Defaults to 1.
            contrastive_weight (float, optional): The weight for the contrastive loss to balance its contribution with the consistency loss. Defaults to 0.2.
        """
        super(TFCLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.contrastive_weight = contrastive_weight

    def _contrastive_loss(self, xi: torch.Tensor, xi_aug: torch.Tensor) -> torch.Tensor:
        """Calculates the contrastive loss for a batch of samples.
        (xi, xi_aug) is a positive pair, (xi, xj_aug) is a negative pair where j != i.

        Args:
            xi (torch.Tensor): Tensor of latent representations of the original samples, shape (batch_size, latent_dim).
            xi_aug (torch.Tensor): Tensor of latent representations of the augmented samples, shape (batch_size, latent_dim).
        Returns:
            torch.Tensor: The average contrastive loss for the batch.
        """

        # treat every sample as an anchor for similarity calculation
        # xi will be an anchor and so will xi_aug
        # this way we can calculate the positive pair similarity for both (xi, xi_aug) and (xi_aug, xi) in one operation
        # final shape will be (2*B, latent_dim)
        x = torch.cat([xi_aug, xi], dim=0)
        x = torch.nn.functional.normalize(x, dim=1)  # normalize the representations
        similarity_matrix = x @ x.T # compute the cosine similarity matrix, shape (2*B, 2*B)
        
        # old method, blows up memory for large batch sizes
        # similarity_matrix = torch.cosine_similarity(
        #     x.unsqueeze(1), x.unsqueeze(0), dim=2
        # )

        # positive pair similarities for (xi, xi_aug)
        left_positives = torch.diag(similarity_matrix, xi.shape[0])
        # positive pair similarities for (xi_aug, xi)
        right_positives = torch.diag(similarity_matrix, -xi.shape[0])
        # combine the positive pairs in the order (xi, xi_aug) -> (xi_aug, xi)
        positives = torch.cat([left_positives, right_positives], dim=0).view(
            similarity_matrix.shape[0], 1
        )
        positives /= self.temperature  # scale by temperature

        # mask out self-pair similarities (xi, xi) and (xi_aug, xi_aug) which are on the diagonal of the similarity matrix
        diagonal_mask = np.eye(similarity_matrix.shape[0])
        # mask out the positive pair (xi, xi_aug) similarities which are upper right diagonal
        left_mask = np.eye(similarity_matrix.shape[0], k=-xi.shape[0])
        # mask out the right positive pair (xi_aug, xi) similarities which are lower left diagonal
        right_mask = np.eye(similarity_matrix.shape[0], k=xi.shape[0])

        # combine the mask and invert it to get the negative pair mask
        mask = (
            torch.from_numpy(1 - (diagonal_mask + left_mask + right_mask))
            .bool()
            .to(xi.device)
        )
        negatives = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
        negatives /= self.temperature  # scale by temperature

        logits = torch.cat((positives, negatives), dim=1)
        labels = torch.zeros(similarity_matrix.shape[0]).to(xi.device).long()

        loss = cross_entropy(logits, labels)
        
        # old mathematical equvialent of cross entropy, but not optimized
        # numerator = torch.exp(positives)
        # denominator = torch.exp(negatives).sum(dim=1) + numerator

        # loss = -torch.log(numerator / denominator)
        # loss = loss.mean()

        return loss

    def _consistency_loss(
        self,
        xt: torch.Tensor,
        xf: torch.Tensor,
        xt_aug: torch.Tensor,
        xf_aug: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the triplet loss to enforce consistency between time and frequency representations.
        (xt, xf) is a positive pair while (xt, xf_aug), (xt_aug, xf), and (xt_aug, xf_aug) are negative pairs.
        the positive pair should be more similar than the negative pairs by a margin.

        Args:
            xt (torch.Tensor): Tensor of latent representations of the original time samples, shape (batch_size, latent_dim).
            xf (torch.Tensor): Tensor of latent representations of the original frequency samples, shape (batch_size, latent_dim).
            xt_aug (torch.Tensor): Tensor of latent representations of the augmented time samples, shape (batch_size, latent_dim).
            xf_aug (torch.Tensor): Tensor of latent representations of the augmented frequency samples, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: The average consistency loss for the batch.
        """
        loss_t_f = self._contrastive_loss(xt, xf)  # positive pair loss (xt, xf)

        # negative pair loss (xt, xf_aug)
        loss_t_f_aug = self._contrastive_loss(xt, xf_aug)

        # negative pair loss (xt_aug, xf)
        loss_t_aug_f = self._contrastive_loss(xt_aug, xf)

        # negative pair loss (xt_aug, xf_aug)
        loss_t_aug_f_aug = self._contrastive_loss(xt_aug, xf_aug)

        loss = torch.tensor(0.0).to(xt.device)
        for pair in [loss_t_f_aug, loss_t_aug_f, loss_t_aug_f_aug]:
            loss += torch.relu(loss_t_f - pair + self.margin)

        return loss

    def forward(
        self,
        ht: torch.Tensor,
        ht_aug: torch.Tensor,
        hf: torch.Tensor,
        hf_aug: torch.Tensor,
        zt: torch.Tensor,
        zf: torch.Tensor,
        zt_aug: torch.Tensor,
        zf_aug: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the total loss as a weighted sum of the contrastive losses for time and frequency representations and the consistency loss between them.

        Args:
            ht (torch.Tensor): Tensor of latent representations of the original time samples before projection, shape (batch_size, channel_dim*sequence_length).
            ht_aug (torch.Tensor): Tensor of latent representations of the augmented time samples before projection, shape (batch_size, channel_dim*sequence_length).
            hf (torch.Tensor): Tensor of latent representations of the original frequency samples before projection, shape (batch_size, channel_dim*sequence_length).
            hf_aug (torch.Tensor): Tensor of latent representations of the augmented frequency samples before projection, shape (batch_size, channel_dim*sequence_length).
            zt (torch.Tensor): Tensor of latent representations of the original time samples after projection, shape (batch_size, latent_dim).
            zf (torch.Tensor): Tensor of latent representations of the original frequency samples after projection, shape (batch_size, latent_dim).
            zt_aug (torch.Tensor): Tensor of latent representations of the augmented time samples after projection, shape (batch_size, latent_dim).
            zf_aug (torch.Tensor): Tensor of latent representations of the augmented frequency samples after projection, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: The total loss for the batch.
        """
        time_loss = self._contrastive_loss(ht, ht_aug)
        frequency_loss = self._contrastive_loss(hf, hf_aug)
        
        # this one is more stable during training
        consistency_loss = self._contrastive_loss(zt, zf)  # just use positive pair loss (zt, zf) for now
        # this one is more consistent with the paper
        # consistency_loss = self._consistency_loss(zt, zf, zt_aug, zf_aug)

        loss = self.contrastive_weight * (time_loss + frequency_loss)
        loss += consistency_loss

        return loss, time_loss.item(), frequency_loss.item(), consistency_loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1024
    encoder_dim = 640
    latent_dim = 128

    ht = torch.randn(batch_size, encoder_dim, device=device)
    ht_aug = torch.randn(batch_size, encoder_dim, device=device)
    hf = torch.randn(batch_size, encoder_dim, device=device)
    hf_aug = torch.randn(batch_size, encoder_dim, device=device)

    zt = torch.randn(batch_size, latent_dim, device=device)
    zf = torch.randn(batch_size, latent_dim, device=device)
    zt_aug = torch.randn(batch_size, latent_dim, device=device)
    zf_aug = torch.randn(batch_size, latent_dim, device=device)
    
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    loss_fn = TFCLoss(contrastive_weight=0.4)
    loss = loss_fn(ht, ht_aug, hf, hf_aug, zt, zf, zt_aug, zf_aug)
    print(loss)
    
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    print(f"Allocated before: {before / 1e9:.2f} GB")
    print(f"Allocated after: {after / 1e9:.2f} GB")
    print(f"Peak during loss: {peak / 1e9:.2f} GB")
