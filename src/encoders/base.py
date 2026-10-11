from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union, Any

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None


class BaseVisionEncoder(ABC):
    """Abstract interface for vision backbone models extracting L2-normalized image embeddings."""

    @abstractmethod
    def encode_image(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """Extract a 1D L2-normalized feature vector z_x for an image."""
        pass

    @abstractmethod
    def encode_batch(self, images: List[Union[Image.Image, str, np.ndarray]]) -> np.ndarray:
        """Extract a 2D matrix of shape (N, d) of L2-normalized features."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality d of the output embedding space."""
        pass


class BaseTextEncoder(ABC):
    """Abstract interface for text backbone models extracting L2-normalized text embeddings."""

    @abstractmethod
    def encode_text(self, text: str, template: str = "{text}") -> np.ndarray:
        """Extract a 1D L2-normalized feature vector t_c for a category or query string."""
        pass

    @abstractmethod
    def encode_categories(self, category_names: List[str], template: str = "{category}") -> np.ndarray:
        """Extract a 2D matrix of shape (|C|, d) of L2-normalized category prototypes."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality d of the output embedding space."""
        pass
