"""Data loaders and vocabulary structures."""

from src.data.category_vocab import CategoryVocabulary
from src.data.commons_dataset import CommonsDataset

try:
    from src.data.commons_downloader import download_real_commons_dataset
    __all__ = ["CategoryVocabulary", "CommonsDataset", "download_real_commons_dataset"]
except ImportError:
    __all__ = ["CategoryVocabulary", "CommonsDataset"]
