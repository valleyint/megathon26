"""Scorer 1: Historical Image Retrieval (k-NN) with Temperature-Scaled Consensus."""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
import numpy as np

from src.stage1.scorers.base import BaseScorer
from src.common.types import ScoredCandidate
from src.data.category_vocab import CategoryVocabulary


class KNNScorer(BaseScorer):
    """Retrieves visual neighbors and computes temperature-scaled consensus voting across historical labels."""

    def __init__(
        self,
        temperature: float = 0.07,
        top_k_neighbors: int = 50,
        vocab: Optional[CategoryVocabulary] = None,
        index_path: Optional[Union[str, Path]] = None,
        labels_path: Optional[Union[str, Path]] = None,
    ):
        self.temperature = temperature
        self.top_k_neighbors = top_k_neighbors
        self.vocab = vocab or CategoryVocabulary()

        # Image index vectors and labels: image_id -> List[int] (category_ids)
        self.image_labels: Dict[int, List[int]] = {}
        self.image_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None  # NumPy fallback
        self.faiss_index = None

        if index_path and Path(index_path).exists() and labels_path and Path(labels_path).exists():
            self.load(index_path, labels_path)

    @property
    def name(self) -> str:
        return "knn"

    def _init_faiss(self, dim: int = 768, index_type: str = "IndexFlatIP"):
        try:
            import faiss
            if index_type == "IndexFlatIP":
                self.faiss_index = faiss.IndexFlatIP(dim)
            else:
                self.faiss_index = faiss.IndexFlatIP(dim)
        except ImportError:
            self.faiss_index = None

    def add_historical_images(
        self,
        vectors: np.ndarray,
        image_ids: List[str],
        category_ids_per_image: List[List[int]],
    ) -> None:
        """Add historical image vectors and their category labels to the index."""
        vectors = vectors.astype(np.float32)
        start_idx = len(self.image_ids)

        if self.faiss_index is None and vectors.shape[0] > 0:
            self._init_faiss(dim=vectors.shape[1])

        if self.faiss_index is not None:
            self.faiss_index.add(vectors)

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        for offset, (img_id, cats) in enumerate(zip(image_ids, category_ids_per_image)):
            internal_idx = start_idx + offset
            self.image_ids.append(img_id)
            self.image_labels[internal_idx] = cats

    def _search_neighbors(self, z_x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k neighbors. Returns (indices, cosine_similarities)."""
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            query = z_x.reshape(1, -1).astype(np.float32)
            actual_k = min(k, self.faiss_index.ntotal)
            similarities, indices = self.faiss_index.search(query, actual_k)
            return indices[0], similarities[0]
        elif self.vectors is not None and len(self.vectors) > 0:
            # Fallback pure NumPy inner product search
            actual_k = min(k, len(self.vectors))
            sims = np.dot(self.vectors, z_x)
            top_idx = np.argpartition(sims, -actual_k)[-actual_k:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return top_idx, sims[top_idx]
        else:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

    def score(self, z_x: np.ndarray, top_k: int = 50) -> List[ScoredCandidate]:
        if (self.faiss_index is None or self.faiss_index.ntotal == 0) and (self.vectors is None or len(self.vectors) == 0):
            return []

        neighbor_indices, similarities = self._search_neighbors(z_x, self.top_k_neighbors)
        if len(neighbor_indices) == 0:
            return []

        # Temperature-scaled softmax weights over retrieved neighbors: exp(cos / tau)
        scaled_sims = similarities / self.temperature
        # Numerical stability shift
        scaled_sims = scaled_sims - np.max(scaled_sims)
        weights = np.exp(scaled_sims)
        total_weight = np.sum(weights) + 1e-12
        normalized_weights = weights / total_weight

        # Accumulate consensus scores per category
        category_scores: Dict[int, float] = {}
        category_exemplars: Dict[int, List[str]] = {}

        for idx, sim, weight in zip(neighbor_indices, similarities, normalized_weights):
            if idx < 0:
                continue
            img_id = self.image_ids[idx]
            for cat_id in self.image_labels.get(idx, []):
                category_scores[cat_id] = category_scores.get(cat_id, 0.0) + weight
                if cat_id not in category_exemplars:
                    category_exemplars[cat_id] = []
                if len(category_exemplars[cat_id]) < 3:
                    category_exemplars[cat_id].append(img_id)

        # Sort categories by consensus score
        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        candidates = []
        for cat_id, score_val in sorted_cats:
            name = self.vocab.get_name(cat_id) or f"Category:unknown_{cat_id}"
            candidates.append(
                ScoredCandidate(
                    category_id=cat_id,
                    name=name,
                    score=float(score_val),
                    source=self.name,
                    metadata={"neighbor_exemplars": category_exemplars.get(cat_id, [])},
                )
            )
        return candidates

    def save(self, index_path: Union[str, Path], labels_path: Union[str, Path]) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(labels_path).parent.mkdir(parents=True, exist_ok=True)

        if self.faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self.faiss_index, str(index_path))
            except ImportError:
                np.save(str(index_path) + ".npy", self.vectors)
        elif self.vectors is not None:
            np.save(str(index_path) + ".npy", self.vectors)

        meta = {
            "image_ids": self.image_ids,
            "image_labels": {str(k): v for k, v in self.image_labels.items()},
        }
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def load(self, index_path: Union[str, Path], labels_path: Union[str, Path]) -> None:
        with open(labels_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.image_ids = meta.get("image_ids", [])
        self.image_labels = {int(k): v for k, v in meta.get("image_labels", {}).items()}

        index_p = Path(index_path)
        if index_p.exists():
            try:
                import faiss
                self.faiss_index = faiss.read_index(str(index_p))
            except (ImportError, Exception):
                pass

        npy_path = Path(str(index_path) + ".npy")
        if npy_path.exists():
            self.vectors = np.load(npy_path)
