from typing import Literal, TypedDict

ActivationName = Literal["relu", "gelu", "silu"]
NormalizerName = Literal["zscore", "minmax"]


class WindowIndex(TypedDict):
    trial_idx: int
    start: int
    end: int


class WindowOpts(TypedDict):
    size: int
    stride: int
