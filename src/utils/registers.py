from torch.nn import GELU, ReLU, SiLU

from src.utils.normalization import MinMaxNormalizer, ZScoreNormalizer

NORMALIZERS = {"zscore": ZScoreNormalizer, "minmax": MinMaxNormalizer}
ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}