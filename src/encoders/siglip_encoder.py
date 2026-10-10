"""SigLIP Vision and Text Encoders using HuggingFace Transformers."""

from __future__ import annotations
from typing import List, Union, Optional
import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.encoders.base import BaseVisionEncoder, BaseTextEncoder


class SigLIPVisionEncoder(BaseVisionEncoder):
    """SigLIP Vision Encoder extracting normalized z_x representations."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "float32",
        mock_dim: Optional[int] = None,
    ):
        self.model_name = model_name
        self.mock_dim = mock_dim
        self._dim = mock_dim or 768

        if mock_dim is not None:
            self.processor = None
            self.model = None
            self.device = "cpu"
            return

        import torch
        from transformers import AutoProcessor, AutoModel

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()

        # Determine embedding dimension from model config
        if hasattr(self.model.config, "vision_config"):
            self._dim = self.model.config.vision_config.hidden_size
        elif hasattr(self.model.config, "hidden_size"):
            self._dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _load_image(self, image: Union[Image.Image, str, np.ndarray]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def encode_image(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        return self.encode_batch([image])[0]

    def encode_batch(self, images: List[Union[Image.Image, str, np.ndarray]]) -> np.ndarray:
        if self.mock_dim is not None:
            # Deterministic mock embedding for unit testing
            vectors = np.random.randn(len(images), self.mock_dim).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12
            return vectors / norms

        import torch

        pil_images = [self._load_image(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # L2 normalization
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.cpu().to(torch.float32).numpy()


class SigLIPTextEncoder(BaseTextEncoder):
    """SigLIP Text Encoder for Zero-Shot text prototypes."""

    DEFAULT_TEMPLATE = "a Wikimedia Commons photograph categorized under {name}"

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "float32",
        mock_dim: Optional[int] = None,
    ):
        self.model_name = model_name
        self.mock_dim = mock_dim
        self._dim = mock_dim or 768

        if mock_dim is not None:
            self.processor = None
            self.model = None
            self.device = "cpu"
            return

        import torch
        from transformers import AutoProcessor, AutoModel

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()

        if hasattr(self.model.config, "text_config"):
            self._dim = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, "hidden_size"):
            self._dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_text(self, text: str, template: Optional[str] = None) -> np.ndarray:
        return self.encode_categories([text], template=template)[0]

    def encode_categories(
        self,
        category_names: List[str],
        template: Optional[str] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        if self.mock_dim is not None:
            vectors = np.random.randn(len(category_names), self.mock_dim).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12
            return vectors / norms

        import torch

        tmpl = template or self.DEFAULT_TEMPLATE
        # Strip "Category:" prefix when formatting into prompt
        cleaned_names = [
            c.replace("Category:", "").strip() if c.startswith("Category:") else c.strip()
            for c in category_names
        ]
        formatted_texts = [tmpl.format(name=c) for c in cleaned_names]

        all_embeddings = []
        for i in range(0, len(formatted_texts), batch_size):
            chunk = formatted_texts[i : i + batch_size]
            inputs = self.processor(text=chunk, padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)
