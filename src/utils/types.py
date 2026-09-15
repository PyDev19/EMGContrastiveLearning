from typing import Literal, TypedDict

ActivationName = Literal["relu", "gelu", "silu"]


class WindowIndex(TypedDict):
    trial_idx: int
    start: int
    end: int


class WindowOpts(TypedDict):
    size: int
    stride: int
