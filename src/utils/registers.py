from torch.nn import GELU, BatchNorm1d, GroupNorm, LayerNorm, ReLU, SiLU

from src.datasets.physiomio import PhysioMioDataset
from src.loss.supcon import SupervisedContrastiveLoss
from src.models.roformer_contrastive import RoFormerConstrastiveModel
from src.utils.normalization import MinMaxNormalizer, ZScoreNormalizer

NORMALIZERS = {"zscore": ZScoreNormalizer, "minmax": MinMaxNormalizer}
ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "silu": SiLU}
NORM_LAYERS = {"batch1d": BatchNorm1d, "layer": LayerNorm, "group": GroupNorm}
DATASETS = {"physiomio": PhysioMioDataset}
MODELS = {"roformer_contrastive": RoFormerConstrastiveModel}
LOSSES = {"supervised_contrastive": SupervisedContrastiveLoss}
