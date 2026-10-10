"""Candidate Aggregation and Fusion for Stage 1."""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np

from src.common.types import ScoredCandidate, Stage1CandidatePool


class Stage1Aggregator:
    """Combines candidates from multiple Stage 1 scorers into a unified candidate pool."""

    def __init__(
        self,
        pool_size: int = 50,
        knn_weight: float = 0.5,
        prototype_weight: float = 0.5,
        supervised_weight: float = 0.5,
    ):
        self.pool_size = pool_size
        self.weights = {
            "knn": knn_weight,
            "prototype": prototype_weight,
            "supervised": supervised_weight,
        }

    def aggregate(
        self,
        query_id: str,
        embedding: np.ndarray,
        scorer_outputs: Dict[str, List[ScoredCandidate]],
        latency_ms: Optional[Dict[str, float]] = None,
    ) -> Stage1CandidatePool:
        """Fuses candidate rankings from multiple scorers.

        Uses Reciprocal Rank Fusion (RRF) combined with weighted scorer signals
        to ensure high recall across both instance consensus and zero-shot semantics.
        """
        combined_scores: Dict[int, float] = {}
        candidate_map: Dict[int, ScoredCandidate] = {}

        # 60 is standard RRF constant (Cormack et al.)
        rrf_k = 60.0

        for source_name, candidate_list in scorer_outputs.items():
            weight = self.weights.get(source_name, 1.0)
            for rank, candidate in enumerate(candidate_list):
                cat_id = candidate.category_id

                # RRF score component
                rrf_score = weight / (rrf_k + rank + 1)
                combined_scores[cat_id] = combined_scores.get(cat_id, 0.0) + rrf_score

                if cat_id not in candidate_map:
                    candidate_map[cat_id] = candidate
                else:
                    # Merge metadata (e.g. keep exemplars if present)
                    if candidate.metadata:
                        candidate_map[cat_id].metadata.update(candidate.metadata)

        # Sort by fused score
        sorted_cat_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_cat_ids = sorted_cat_ids[: self.pool_size]

        final_candidates: List[ScoredCandidate] = []
        for cat_id, fused_score in top_cat_ids:
            base_cand = candidate_map[cat_id]
            final_candidates.append(
                ScoredCandidate(
                    category_id=cat_id,
                    name=base_cand.name,
                    score=float(fused_score),
                    source="stage1_fused",
                    metadata=base_cand.metadata,
                )
            )

        return Stage1CandidatePool(
            query_id=query_id,
            embedding=embedding,
            candidates=final_candidates,
            scorer_breakdown=scorer_outputs,
            latency_ms=latency_ms or {},
        )
