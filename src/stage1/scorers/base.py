"""Base class for Stage 1 candidate scorers."""

from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from src.common.types import ScoredCandidate


class BaseScorer(ABC):
    """Abstract interface for all Stage 1 candidate scorers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the scorer (e.g. 'knn', 'prototype', 'supervised')."""
        pass

    @abstractmethod
    def score(self, z_x: np.ndarray, top_k: int = 50) -> List[ScoredCandidate]:
        """Compute category candidate scores for a single query vector z_x.

        Args:
            z_x: 1D L2-normalized query vector of shape (d,).
            top_k: Number of top candidate categories to retrieve.

        Returns:
            List of ScoredCandidate instances sorted by descending score.
        """
        pass

    def score_batch(self, Z: np.ndarray, top_k: int = 50) -> List[List[ScoredCandidate]]:
        """Batch scoring implementation (default iterates over rows)."""
        return [self.score(z, top_k=top_k) for z in Z]
