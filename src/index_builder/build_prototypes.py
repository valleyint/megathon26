"""Offline builder for Zero-Shot Category Prototypes."""

from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np

from src.data.category_vocab import CategoryVocabulary
from src.encoders.base import BaseTextEncoder


def build_category_prototypes(
    vocab: CategoryVocabulary,
    text_encoder: BaseTextEncoder,
    output_path: Union[str, Path],
    batch_size: int = 64,
) -> np.ndarray:
    """Encodes all categories in the vocabulary using the text encoder and saves to disk."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    all_names = vocab.get_all_names()
    print(f"Encoding {len(all_names)} categories with {text_encoder.__class__.__name__}...")

    prototypes = text_encoder.encode_categories(all_names, batch_size=batch_size)
    np.save(out_p, prototypes)
    print(f"Saved prototype matrix of shape {prototypes.shape} to {out_p}")
    return prototypes
