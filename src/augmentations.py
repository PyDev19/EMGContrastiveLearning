import numpy as np
import torch


class Augmentations:
    def __init__(
        self,
        jitter_sigma=0.8,
        scale_sigma=1.1,
        mask_prob=0.5,
        possible_num_segments=[2, 4, 8, 16],
        freq_perturb_ratio=0.25,
        freq_alpha=0.1,
    ):
        """Initialize the Augmentations class with specified parameters for each augmentation technique.

        Args:
            jitter_sigma (float, optional): Standard deviation for jitter augmentation. Defaults to 0.8.
            scale_sigma (float, optional): Standard deviation for scale augmentation. Defaults to 1.1.
            mask_prob (float, optional): Probability of masking segments. Defaults to 0.5.
            possible_num_segments (list, optional): List of possible number of segments for permutation augmentation. Defaults to [2, 4, 5].
            freq_perturb_ratio (float, optional): Ratio of frequencies to perturb. Defaults to 0.25.
            freq_alpha (float, optional): Alpha parameter for frequency perturbation. Defaults to 0.1.
        """
        
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.mask_prob = mask_prob
        self.possible_num_segments = possible_num_segments
        self.freq_perturb_ratio = freq_perturb_ratio
        self.freq_alpha = freq_alpha

    def _jitter(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise to the input EMG window.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The jittered EMG window, with the same shape as the input.
        """
        
        noise = torch.randn_like(emg_window) * self.jitter_sigma
        return emg_window + noise

    def _scale(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Scales the EMG window

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The scaled EMG window, with the same shape as the input.
        """
        factor = torch.normal(mean=1.0, std=self.scale_sigma, size=emg_window.shape)
        return emg_window * factor

    def _permutation(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Randomly permute segments of the input EMG window. Segments are defined by splitting the window into a random number of equal parts, which are then permuted.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The permuted EMG window, with the same shape as the input.
        """
        
        orig_steps = np.arange(emg_window.shape[1])
        num_segments = np.random.choice(self.possible_num_segments)

        segment_size = emg_window.shape[1] // num_segments
        split_points = np.arange(1, num_segments) * segment_size
        splits = np.split(orig_steps[: num_segments * segment_size], split_points)

        warp = np.concatenate(np.random.permutation(splits)).ravel()
        permuted_window = emg_window[:, warp]

        return permuted_window

    def _mask(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Randomly mask out segments of the input EMG window by setting them to zero. The masking is applied with a specified probability.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The masked EMG window, with the same shape as the input.
        """
        
        nan_mask = ~emg_window.isnan().any(dim=-1)
        emg_window[~nan_mask] = 0

        mask = torch.from_numpy(
            np.random.binomial(1, self.mask_prob, size=emg_window.shape)
        ).to(torch.bool)
        emg_window[mask] = 0

        return emg_window
    
    def _zero_frequency(self, emg_fft: torch.Tensor) -> torch.Tensor:
        """Randomly zero a select few frequencies in the given FFT of a EMG window

        Args:
            emg_fft (torch.Tensor): The input FFT of an EMG window, with shape (channel_dim, sequence_length) and complex dtype.

        Returns:
            torch.Tensor: The augmented FFT with certain frequencies zeroed out, maintaining the same shape and dtype as the input.
        """
        
        mask = torch.FloatTensor(emg_fft.shape).uniform_() > self.freq_perturb_ratio
        mask = mask.to(emg_fft.device)
        return emg_fft * mask
    
    def _add_frequency(self, emg_fft: torch.Tensor) -> torch.Tensor:
        """Randomly selects frequencies in the given FFT that are lower than the maximum amplitude multiplied by a specified alpha, and sets them to the maximum amplitude multiplied by that alpha.

        Args:
            emg_fft (torch.Tensor): The input FFT of an EMG window, with shape (channel_dim, sequence_length) and complex dtype.

        Returns:
            torch.Tensor: The augmented FFT with added frequency perturbations, maintaining the same shape and dtype as the input.
        """
        mask = torch.FloatTensor(emg_fft.shape).uniform_() > (1- self.freq_perturb_ratio)
        mask = mask.to(emg_fft.device)
        
        max_amp = emg_fft.max()
        random_amp = torch.rand(mask.shape)*(max_amp*self.freq_alpha)
        perturb_matrix = mask * random_amp
        
        return emg_fft + perturb_matrix

    def time_augment(self, emg_window: torch.Tensor) -> torch.Tensor:
        """Apply a random combination of jitter, scale, permutation, and masking augmentations to the input EMG window. At least one augmentation is guaranteed to be applied.

        Args:
            emg_window (torch.Tensor): The input EMG window, with shape (channel_dim, sequence_length).

        Returns:
            torch.Tensor: The augmented EMG window, with the same shape as the input.
        """
        augmentations = [self._jitter, self._scale, self._mask]
        np.random.shuffle(augmentations)

        applied = False
        for aug in augmentations:
            if np.random.rand() < 0.5:
                emg_window = aug(emg_window)
                applied = True

        if not applied:
            emg_window = augmentations[0](emg_window)

        return emg_window
    
    def frequency_augment(self, emg_fft: torch.Tensor) -> torch.Tensor:
        """Apply a random combination of zeroing and adding frequencies to the input FFT of the EMG. At least one augmentation is guaranteed to be applied.

        Args:
            emg_fft (torch.Tensor): The input FFT of an EMG window, with shape (channel_dim, sequence_length) and complex dtype.

        Returns:
            torch.Tensor: The augmented FFT with added frequency perturbations, maintaining the same shape and dtype as the input.
        """
        
        augmentations = [self._zero_frequency, self._add_frequency]
        np.random.shuffle(augmentations)

        applied = False
        for aug in augmentations:
            if np.random.rand() < 0.5:
                emg_fft = aug(emg_fft)
                applied = True

        if not applied:
            emg_fft = augmentations[0](emg_fft)

        return emg_fft

    def __call__(self, emg_window: torch.Tensor, emg_fft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.time_augment(emg_window), self.frequency_augment(emg_fft)