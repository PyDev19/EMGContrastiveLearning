from typing import TypedDict


class WindowIndex(TypedDict):
    trial_idx: int
    start: int
    end: int

class WindowOpts(TypedDict):
    size: int
    stride: int
    