from typing import Literal, TypedDict, TypeVar

T = TypeVar("T")

ActivationName = Literal["relu", "gelu", "silu"]
NormalizerName = Literal["zscore", "minmax"]
NormalizerScopeName = Literal["global", "per-patient"]
NormLayerName = Literal["layer", "rms"]
PoolName = Literal["mean", "attention"]
ModelName = Literal[
    "roformer_time_contrastive", "roformer_time_dann_contrastive"
]
DatasetName = Literal["physiomio", "bep"]
TaskName = Literal["time_contrastive", "time_dann_contrastive"]
LossName = Literal["supervised_contrastive"]


class WindowIndex(TypedDict):
    trial_idx: int
    start: int
    end: int


class WindowOpts(TypedDict):
    size: int
    stride: int
