"""Topological Category DAG Ancestor Suppression enforcing Commons 'most-specific' cataloging rule."""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional

from src.common.types import ScoredCandidate, PruningDecision
from src.stage2.graph.category_dag import CategoryDAG


class AncestorSuppressor:
    """Suppresses broad parent categories when high-confidence descendants are present in the candidate pool."""

    def __init__(
        self,
        dag: CategoryDAG,
        child_confidence_threshold: float = 0.35,
        min_retained_candidates: int = 5,
        max_depth: int = 12,
    ):
        self.dag = dag
        self.child_confidence_threshold = child_confidence_threshold
        self.min_retained_candidates = min_retained_candidates
        self.max_depth = max_depth

    def suppress(
        self,
        candidates: List[ScoredCandidate],
    ) -> Tuple[List[ScoredCandidate], List[PruningDecision]]:
        """Applies DAG ancestor pruning over the candidate pool."""
        if not candidates:
            return [], []

        cand_map: Dict[int, ScoredCandidate] = {c.category_id: c for c in candidates}
        pool_ids: Set[int] = set(cand_map.keys())

        suppressed_ids: Set[int] = set()
        decisions: List[PruningDecision] = []

        # For each candidate c in C_pool:
        for c in candidates:
            c_id = c.category_id
            # Query descendants of c in the DAG
            descendants = self.dag.get_descendants(c_id, max_depth=self.max_depth)
            pool_descendants = descendants & pool_ids

            # Check if any descendant d has confidence score s(d) >= tau_child
            confident_children = [
                cand_map[d_id]
                for d_id in pool_descendants
                if cand_map[d_id].score >= self.child_confidence_threshold
            ]

            if confident_children:
                # Find the most confident child triggering the suppression
                best_child = max(confident_children, key=lambda x: x.score)
                suppressed_ids.add(c_id)
                decisions.append(
                    PruningDecision(
                        category_id=c_id,
                        name=c.name,
                        status="suppressed_ancestor",
                        score=c.score,
                        triggering_descendant=best_child.name,
                        reason=(
                            f"Suppressed broad ancestor category because specific descendant '{best_child.name}' "
                            f"is present with confidence {best_child.score:.3f} >= {self.child_confidence_threshold}"
                        ),
                        metadata={
                            "triggering_child_id": best_child.category_id,
                            "triggering_child_name": best_child.name,
                            "triggering_child_score": best_child.score,
                            "threshold": self.child_confidence_threshold,
                        },
                    )
                )

        # Safety fallback: ensure at least min_retained_candidates remain
        retained_candidates = [c for c in candidates if c.category_id not in suppressed_ids]

        if len(retained_candidates) < self.min_retained_candidates and suppressed_ids:
            # Unsuppress highest-scoring suppressed ancestors until min threshold is reached
            needed = self.min_retained_candidates - len(retained_candidates)
            suppressed_list = sorted(
                [c for c in candidates if c.category_id in suppressed_ids],
                key=lambda x: x.score,
                reverse=True,
            )
            unsuppressed = set()
            for cand in suppressed_list[:needed]:
                unsuppressed.add(cand.category_id)
                retained_candidates.append(cand)

            # Update decisions for unsuppressed items
            for d in decisions:
                if d.category_id in unsuppressed:
                    d.status = "retained_fallback"
                    d.reason = "Unsuppressed to maintain minimum candidate pool budget"

        # Record retained decisions for clarity
        retained_ids = {c.category_id for c in retained_candidates}
        for c in retained_candidates:
            if not any(d.category_id == c.category_id for d in decisions):
                decisions.append(
                    PruningDecision(
                        category_id=c.category_id,
                        name=c.name,
                        status="retained",
                        score=c.score,
                        reason="Retained: Specific leaf or broad fallback without triggering descendant",
                    )
                )

        # Sort retained candidates by score descending
        retained_candidates.sort(key=lambda x: x.score, reverse=True)

        return retained_candidates, decisions
