from typing import Literal, TypedDict, TypeVar

T = TypeVar("T")

ActivationName = Literal["relu", "gelu", "silu"]
NormalizerName = Literal["zscore", "minmax"]
ModelName = Literal[
    "roformer_time_contrastive", "roformer_time_domain_adversial_contrastive"
]
DatasetName = Literal["physiomio", "bep"]
TaskName = Literal["time_contrastive", "time_domain_adversial_contrastive"]
LossName = Literal["supervised_contrastive"]


class WindowIndex(TypedDict):
    trial_idx: int
    start: int
    end: int


class WindowOpts(TypedDict):
    size: int
    stride: int
