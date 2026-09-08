from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(eq=False)
class Augmentations:
    """Initialize the Augmentations class with specified parameters for each augmentation technique.

    Args:
        jitter_sigma (float, optional): Standard deviation for jitter augmentation. Defaults to 0.8.
        scale_sigma (float, optional): Standard deviation for scale augmentation. Defaults to 1.1.
        mask_size (list[int], optional): Size [C, T] of masked patches where C <= sEMG channels and T <= sEMG timesteps. Defaults to [4, 32].
        mask_prob (float, optional): Probability of masking segments. Defaults to 0.5.
        possible_num_segments (list, optional): List of possible number of segments for permutation augmentation. Defaults to [2, 4, 5].
        freq_perturb_ratio (float, optional): Ratio of frequencies to perturb. Defaults to 0.25.
        freq_alpha (float, optional): Alpha parameter for frequency perturbation. Defaults to 0.1.
    """

    jitter_sigma: float = 0.8
    scale_sigma: float = 1.1
    mask_size: list = field(default_factory=lambda: [4, 32])
    mask_prob: float = 0.5
    freq_perturb_ratio: float = 0.25
    freq_alpha: float = 0.1

    def jitter(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise to the input EMG window.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The jittered EMG window, with the same shape as the input.
        """

        noise = torch.randn_like(emg_window) * self.jitter_sigma
        return emg_window + noise

    def scale(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Scales the EMG window

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The scaled EMG window, with the same shape as the input.
        """
        factor = torch.normal(mean=1.0, std=self.scale_sigma, size=emg_window.shape)
        return emg_window * factor

    def patch_mask(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Randomly mask out square patches of the input EMG window by setting them to zero. The masking is applied with a specified probability.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The masked EMG window, with the same shape as the input.
        """

        C, T = emg_window.size()
        patch_height, patch_width = self.mask_size

        # calculate the total number of patches
        num_patches = (C // patch_height) * (T // patch_width)
        masked_patches = int(num_patches * self.mask_prob)

        # calculate coordinates for all possible patches within [C, T]
        row_indices = torch.arange(0, C, patch_height)
        col_indices = torch.arange(0, T, patch_width)
        patches = [(i, j) for i in row_indices for j in col_indices]

        # select top masked_patches of random permutations of ints from 0 to num_patches-1
        patch_indices = torch.randperm(num_patches)[:masked_patches]
        mask = torch.zeros(C, T, dtype=torch.bool)  # generate mask of size [C, T]

        # select patches and apply them to the mask
        for i in patch_indices:
            row, col = patches[i]
            mask[row : row + patch_height, col : col + patch_width] = True

        masked_emg_window = emg_window.clone()
        masked_emg_window[mask] = 0

        return masked_emg_window

    def time_augment(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Apply a random combination of jitter, scale, permutation, and masking augmentations to the input EMG window. At least one augmentation is guaranteed to be applied.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The augmented EMG window, with the same shape as the input.
        """
        augmentations = np.array([self.jitter, self.scale, self.patch_mask])
        np.random.shuffle(augmentations)

        applied = False
        for aug in augmentations:
            if np.random.rand() < 0.5:
                emg_window = aug(emg_window)
                applied = True

        if not applied:
            emg_window = augmentations[0](emg_window)

        return emg_window

    def __call__(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Apply a random combination of jitter, scale, permutation, and masking augmentations to the input EMG window. At least one augmentation is guaranteed to be applied.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The augmented EMG window, with the same shape as the input.
        """
        return self.time_augment(emg_window)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    augmentations = Augmentations(mask_prob=0.25)
    sample_time = torch.rand(64, 512) * 255
    masked_time = augmentations.time_augment(sample_time)

    print(sample_time)
    print(masked_time)

    img1 = sample_time.cpu().numpy()
    img2 = masked_time.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img1, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("Masked")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
