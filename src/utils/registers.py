from collections.abc import Callable

from torch.nn import GELU, ReLU, SiLU

from src.utils.normalization import MinMaxNormalizer, ZScoreNormalizer
from src.utils.types import DatasetName, LossName, ModelName, T

NORMALIZERS = {"zscore": ZScoreNormalizer, "minmax": MinMaxNormalizer}
ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}
MODELS = {}
DATASETS = {}
LOSSES = {}


def register(
    registry: dict[str, type], name: ModelName | DatasetName | LossName | None = None
) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        key = name if name is not None else cls.__name__

        if key in registry:
            raise ValueError(
                f"'{key}' is already registered in this registry "
                f"(existing: {registry[key]!r}, new: {cls!r})"
            )

        registry[key] = cls
        return cls

    return decorator
