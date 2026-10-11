"""Scorer 2: Zero-Shot Text-Prototype Cosine Matching."""

from __future__ import annotations
from typing import List, Optional, Union
try:
    import numpy as np
except ImportError:
    np = None

from src.stage1.scorers.base import BaseScorer
from src.common.types import ScoredCandidate
from src.data.category_vocab import CategoryVocabulary


class PrototypeScorer(BaseScorer):
    """Zero-shot prototype matching scorer against precomputed category representations."""

    def __init__(
        self,
        prototypes: Optional[np.ndarray] = None,
        vocab: Optional[CategoryVocabulary] = None,
        prototypes_path: Optional[Union[str, Path]] = None,
        vocab_path: Optional[Union[str, Path]] = None,
    ):
        self.vocab = vocab or CategoryVocabulary()
        if vocab_path and Path(vocab_path).exists():
            self.vocab = CategoryVocabulary.load(vocab_path)

        if prototypes is not None:
            self.prototypes = prototypes.astype(np.float32)
        elif prototypes_path and Path(prototypes_path).exists():
            self.prototypes = np.load(prototypes_path).astype(np.float32)
        else:
            self.prototypes = np.empty((0, 768), dtype=np.float32)

    @property
    def name(self) -> str:
        return "prototype"

    def score(self, z_x: np.ndarray, top_k: int = 50) -> List[ScoredCandidate]:
        if len(self.prototypes) == 0:
            return []

        # Vector-matrix multiplication: shape (|C|,)
        similarities = np.dot(self.prototypes, z_x)

        k = min(top_k, len(similarities))
        if k <= 0:
            return []

        # Efficient top-k selection via argpartition
        if k < len(similarities):
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        else:
            top_indices = np.argsort(-similarities)

        results = []
        for idx in top_indices:
            cat_id = int(idx)
            name = self.vocab.get_name(cat_id) or f"Category:unknown_{cat_id}"
            results.append(
                ScoredCandidate(
                    category_id=cat_id,
                    name=name,
                    score=float(similarities[idx]),
                    source=self.name,
                )
            )
        return results

    def add_category(self, name: str, text_vector: np.ndarray) -> int:
        """Continual learning: append a newly created Commons category prototype in memory."""
        # Normalize incoming vector
        norm = np.linalg.norm(text_vector)
        normed_vec = (text_vector / (norm + 1e-12)).astype(np.float32)

        cat_id = self.vocab.add_category(name)
        if len(self.prototypes) == 0:
            self.prototypes = normed_vec.reshape(1, -1)
        else:
            self.prototypes = np.vstack([self.prototypes, normed_vec.reshape(1, -1)])
        return cat_id

    def save(self, prototypes_path: Union[str, Path], vocab_path: Optional[Union[str, Path]] = None) -> None:
        Path(prototypes_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(prototypes_path, self.prototypes)
        if vocab_path:
            self.vocab.save(vocab_path)
