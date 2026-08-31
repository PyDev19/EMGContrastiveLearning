from abc import ABC, abstractmethod

import torch


class Normalizer(ABC):
    def __init__(self, eps: float = 1e-08):
        """Initialize the normalizer.

        Args:
            eps: Small constant added to the denominator in ``transform``
                to avoid division by zero for constant-valued channels.
        """
        self.eps = eps
        self._fitted = False

    def _check_fitted(self):
        """Raise if ``transform`` is called before ``fit``.

        Raises:
            RuntimeError: If ``fit`` has not been called on this instance.
        """
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}: fit() must be called before transform()"
            )

    @abstractmethod
    def fit(self, x: torch.Tensor) -> None: ...

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Runs ``fit`` if the normalizer hasn't been fit yet and then runs ``transform``
        to normalize the given tensor.

        Args:
            x: Tensor to normalize (will be used to fit if the normalizer hasn't been fit yet)

        Returns:
            torch.Tensor: The normalized tensor, same shape as ``x``
        """
        if not self._fitted:
            self.fit(x)

        return self.transform(x)

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit statistics on ``x`` and immediately apply them to ``x``.

        Convenience method equivalent to calling ``fit`` followed by
        ``transform`` on the same tensor.

        Args:
            x: Tensor to both fit on and normalize.

        Returns:
            torch.Tensor: The normalized tensor, same shape as ``x``.
        """
        self.fit(x)
        return self.transform(x)

    def to(self, device: str | torch.device):
        """Move all tensor-valued statistics to the given device, in place.

        Use this when statistics were fit on one device (e.g. CPU, during
        preprocessing) but need to be applied to tensors on another device
        (e.g. GPU, during training).

        Args:
            device: Target device, as accepted by ``torch.Tensor.to``.
        """
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))

    def state_dict(self) -> dict:
        """Return a copy of this normalizer's internal state.

        Includes fitted statistics, ``eps``, and the fitted flag, so a
        normalizer's exact behavior can be reproduced later via
        ``load_state_dict`` on a freshly constructed instance.

        Returns:
            A shallow copy of this instance's ``__dict__``.
        """
        return dict(vars(self))

    def load_state_dict(self, state_dict: dict):
        """Restore internal state previously produced by ``state_dict``.

        Args:
            state_dict: A dict as returned by ``state_dict``, typically
                loaded from a checkpoint.
        """
        for k, v in state_dict.items():
            setattr(self, k, v)


class ZScoreNormalizer(Normalizer):
    """Normalizes by subtracting the mean and dividing by the standard deviation.

    Statistics (``mean`` and ``std``) are computed per-channel, pooling over
    the trial and time dimensions, assuming inputs of shape
    ``(trials, channels, time)``. The resulting ``(channels,)`` statistics
    are reshaped to ``(1, channels, 1)`` so they broadcast correctly against
    inputs of that shape during ``transform``.
    """

    def __init__(self, eps: float = 1e-08):
        super().__init__(eps=eps)

    def fit(self, x: torch.Tensor):
        """Compute per-channel mean and std, pooling over trials and time.

        Args:
            x: Reference tensor of shape ``(trials, channels, time)`` to
                compute statistics from (typically the training split).
        """
        self.mean = torch.mean(x, dim=(0, 2)).reshape(1, -1, 1)
        self.std = torch.std(x, dim=(0, 2)).reshape(1, -1, 1)
        self._fitted = True

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization: ``(x - mean) / (std + eps)``.

        Args:
            x: Tensor of shape ``(trials, channels, time)`` to normalize.
                Must have the same number of channels as the tensor passed
                to ``fit``.

        Returns:
            torch.Tensor: The normalized tensor, same shape as ``x``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return (x - self.mean) / (self.std + self.eps)


class MinMaxNormalizer(Normalizer):
    """Normalizes values to the [0, 1] range using min/max statistics.

    Statistics (``min`` and ``max``) are computed per-channel, pooling over
    the trial and time dimensions, assuming inputs of shape
    ``(trials, channels, time)``. The resulting ``(channels,)`` statistics
    are reshaped to ``(1, channels, 1)`` so they broadcast correctly against
    inputs of that shape during ``transform``.
    """

    def __init__(self, eps: float = 1e-08):
        super().__init__(eps=eps)

    def fit(self, x: torch.Tensor):
        """Compute per-channel min and max, pooling over trials and time.

        Args:
            x: Reference tensor of shape ``(trials, channels, time)`` to
                compute statistics from (typically the training split).
        """
        self.max = torch.amax(x, dim=(0, 2)).reshape(1, -1, 1)
        self.min = torch.amin(x, dim=(0, 2)).reshape(1, -1, 1)
        self._fitted = True

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization: ``(x - min) / (max - min + eps)``.

        Args:
            x: Tensor of shape ``(trials, channels, time)`` to normalize.
                Must have the same number of channels as the tensor passed
                to ``fit``.

        Returns:
            torch.Tensor: The normalized tensor, same shape as ``x``.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._check_fitted()
        return (x - self.min) / (self.max - self.min + self.eps)
