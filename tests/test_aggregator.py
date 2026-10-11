"""Unit tests for Stage 1 Aggregator."""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import numpy as np
except ImportError:
    np = None

from src.common.types import ScoredCandidate
from src.stage1.aggregator import Stage1Aggregator


class TestAggregator(unittest.TestCase):
    def test_aggregation_and_fusion(self):
        aggregator = Stage1Aggregator(pool_size=3)

        # Scorer outputs
        knn_candidates = [
            ScoredCandidate(category_id=1, name="Category:A", score=0.9, source="knn", metadata={"exemplars": ["img1"]}),
            ScoredCandidate(category_id=2, name="Category:B", score=0.5, source="knn"),
        ]
        proto_candidates = [
            ScoredCandidate(category_id=2, name="Category:B", score=0.8, source="prototype"),
            ScoredCandidate(category_id=3, name="Category:C", score=0.6, source="prototype"),
            ScoredCandidate(category_id=4, name="Category:D", score=0.4, source="prototype"),
        ]

        dummy_emb = np.array([1.0, 0.0], dtype=np.float32) if np is not None else [1.0, 0.0]
        pool = aggregator.aggregate(
            query_id="q1",
            embedding=dummy_emb,
            scorer_outputs={"knn": knn_candidates, "prototype": proto_candidates},
            latency_ms={"total_ms": 15.0},
        )

        # Pool size capped at 3
        self.assertEqual(len(pool.candidates), 3)
        self.assertEqual(pool.query_id, "q1")
        self.assertEqual(pool.latency_ms["total_ms"], 15.0)

        # Unique category IDs
        cat_ids = pool.category_ids
        self.assertEqual(len(cat_ids), len(set(cat_ids)))

        # Category B was present in both, should have high fused rank
        self.assertIn(2, cat_ids)
        # Category A was rank 0 in KNN, should also be in top
        self.assertIn(1, cat_ids)


if __name__ == "__main__":
    unittest.main()
