import torch
from torch.nn import Module


class SupervisedContrastiveLoss(Module):
    def __init__(self, temperature: float):
        """Supervised Contrastive Loss (Khosla et al., https://arxiv.org/abs/2004.11362).
        L_out formulation, specialized to exactly two views per sample: `z` and an
        augmented view `z_aug`. Concatenating the two views guarantees every anchor
        has at least one positive (its own augmented twin)

        Args:
            temperature (float): value to scale similarity scores for gradient stability
        """

        super().__init__()

        self.temperature = temperature

    def forward(
        self, z: torch.Tensor, labels: torch.Tensor, z_aug: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Vectorized calculation of L_out supcon loss using mean reduction.

        The inner logarithm is expanded via the identity:

            log( exp(z_i . z_p / t) / sum_{a != i} exp(z_i . z_a / t) )
            = (z_i . z_p) / t - log sum_{a != i} exp(z_i . z_a / t)

        where z_i is the anchor, z_p a positive, z_a any other sample in
        the batch, and t is the temperature.

        Args:
            z (torch.Tensor): embedding vectors for non-augmented samples
            z_aug (torch.Tensor): embedding vectors for augmented samples
            labels (torch.Tensor): labels for each sample, should be the same for both views

        Returns:
            torch.Tensor: the supervised contrastive loss for the batch
        """
        device = z.device

        # stack the two views into one batch, so each sample's augmented
        # twin acts as a guaranteed positive for it: (2B, D) / (2B,)
        if z_aug is not None:
            embeddings = torch.cat([z, z_aug], dim=0)
            labels = torch.cat([labels, labels], dim=0).unsqueeze(1)
        else:
            embeddings = z
            labels = labels.unsqueeze(1)

        # cosine similarity between every pair of samples, scaled by temperature
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        similarity_matrix = (embeddings @ embeddings.mT) / self.temperature  # (2B, 2B)

        # anchor is never compared to itself, so mask for them
        # shape (2B, 2B)
        diagonal_mask = torch.eye(
            *similarity_matrix.shape, device=device, dtype=torch.bool
        )

        # mask out diagonals with -inf, shape: (2B, 2B)
        similarity_matrix = similarity_matrix.masked_fill(diagonal_mask, float("-inf"))

        # mask for (anchor, positive) similarity pair not including diagonal
        # shape (2B, 2B)
        positive_mask = torch.eq(labels, labels.mT) & ~diagonal_mask

        # number of positives per anchor
        # atleast one positive sample is guaranteed due to augmented view
        # clamp for safety, shape: (2B)
        num_positives = positive_mask.sum(dim=1).clamp(min=1)

        # denominator sums over all samples except the anchor itself,
        # computed in log-space via logsumexp for numerical stability
        # shape: (2B, 1)
        log_denominator = torch.logsumexp(similarity_matrix, dim=1).unsqueeze(1)

        # log p(j | anchor) for every (anchor, j) pair, via the identity
        # Then zero out non-positive j's with a mask so the row-sum below
        # only accumulates over each anchor's positives.
        log_prob = similarity_matrix - log_denominator
        log_prob_sum = log_prob.masked_fill(~positive_mask, 0).sum(dim=1)  # (2B,)

        #  negative mean log-likelihood of positives per anchor
        loss = (-log_prob_sum / num_positives).mean()

        return loss


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, embed_dim, num_classes = 8, 16, 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = SupervisedContrastiveLoss(temperature=0.07)

    z = torch.randn(batch_size, embed_dim, device=device)
    z_aug = torch.randn(batch_size, embed_dim, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    loss = loss_fn(z, z_aug, labels)
    print(f"random embeddings loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "loss should be finite"

    # sanity check: if every sample's positives (same class + its own
    # augmented twin) are pulled to identical embeddings and everything
    # else is pushed far apart, loss should approach 0
    z_perfect = torch.zeros(batch_size, embed_dim, device=device)
    labels_perfect = torch.arange(
        batch_size, device=device
    )  # every sample its own class
    for i in range(batch_size):
        z_perfect[i, i % embed_dim] = 10.0  # orthogonal-ish per class, large magnitude
    z_aug_perfect = z_perfect.clone()  # augmented view identical to original

    loss_perfect = loss_fn(z_perfect, z_aug_perfect, labels_perfect)
    print(f"near-ideal embeddings loss: {loss_perfect.item():.4f}")
    assert loss_perfect.item() < loss.item(), (
        "well-separated embeddings should have lower loss"
    )

    # gradient check: loss should be differentiable w.r.t. inputs
    z_grad = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
    z_aug_grad = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
    loss_grad = loss_fn(z_grad, z_aug_grad, labels)
    loss_grad.backward()
    assert z_grad.grad is not None and torch.isfinite(z_grad.grad).all(), (
        "gradients should be finite"
    )
    print("gradient check passed")

    # edge case: batch size of 1 (only positive is the augmented twin)
    z_single = torch.randn(1, embed_dim, device=device)
    z_aug_single = torch.randn(1, embed_dim, device=device)
    labels_single = torch.zeros(1, dtype=torch.long, device=device)
    loss_single = loss_fn(z_single, z_aug_single, labels_single)
    assert torch.isfinite(loss_single), "batch size of 1 should not produce NaN/inf"
    print(f"batch size 1 loss: {loss_single.item():.4f}")

    print("all checks passed")
