from torch.nn import GELU, BatchNorm1d, GroupNorm, LayerNorm, ReLU, SiLU

from src.utils import MinMaxNormalizer, ZScoreNormalizer

NORMALIZERS = {"zscore": ZScoreNormalizer, "minmax": MinMaxNormalizer}
ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}
NORM_LAYERS = {"batch": BatchNorm1d, "layer": LayerNorm, "group": GroupNorm}
