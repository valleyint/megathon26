"""Unit tests for AncestorSuppressor."""

import unittest
from src.common.types import ScoredCandidate
from src.stage2.graph.category_dag import CategoryDAG
from src.stage2.graph.ancestor_suppressor import AncestorSuppressor


class TestAncestorSuppressor(unittest.TestCase):
    def setUp(self):
        self.dag = CategoryDAG()
        # 0: Architecture (Broad root)
        # 1: Gothic architecture (Mid-level)
        # 2: Gothic cathedrals (Specific descendant)
        # 3: Eiffel Tower (Unrelated)
        # 4: Eiffel Tower at night (Specific descendant of Eiffel Tower)
        self.dag.add_category(0, "Category:Architecture")
        self.dag.add_category(1, "Category:Gothic architecture")
        self.dag.add_category(2, "Category:Gothic cathedrals")
        self.dag.add_category(3, "Category:Eiffel Tower")
        self.dag.add_category(4, "Category:Eiffel Tower at night")

        self.dag.add_edge(0, 1)
        self.dag.add_edge(1, 2)
        self.dag.add_edge(3, 4)

        self.suppressor = AncestorSuppressor(
            dag=self.dag,
            child_confidence_threshold=0.35,
            min_retained_candidates=2,
        )

    def test_suppress_ancestor_when_child_is_confident(self):
        # Child 2 has high confidence (0.85 >= 0.35)
        # Both ancestors 0 and 1 should be suppressed!
        candidates = [
            ScoredCandidate(category_id=0, name="Category:Architecture", score=0.90, source="stage1"),
            ScoredCandidate(category_id=1, name="Category:Gothic architecture", score=0.88, source="stage1"),
            ScoredCandidate(category_id=2, name="Category:Gothic cathedrals", score=0.85, source="stage1"),
            ScoredCandidate(category_id=3, name="Category:Eiffel Tower", score=0.40, source="stage1"),
        ]

        retained, decisions = self.suppressor.suppress(candidates)
        retained_ids = [c.category_id for c in retained]

        # 0 and 1 must be suppressed in favor of 2
        self.assertNotIn(0, retained_ids)
        self.assertNotIn(1, retained_ids)
        self.assertIn(2, retained_ids)
        self.assertIn(3, retained_ids)

        suppressed_decisions = [d for d in decisions if d.status == "suppressed_ancestor"]
        suppressed_ids = {d.category_id for d in suppressed_decisions}
        self.assertEqual(suppressed_ids, {0, 1})

    def test_retain_ancestor_when_child_confidence_is_low(self):
        # Child 2 has low confidence (0.20 < 0.35)
        # Parent 1 should be retained as safe broad fallback!
        candidates = [
            ScoredCandidate(category_id=1, name="Category:Gothic architecture", score=0.80, source="stage1"),
            ScoredCandidate(category_id=2, name="Category:Gothic cathedrals", score=0.20, source="stage1"),
        ]

        retained, decisions = self.suppressor.suppress(candidates)
        retained_ids = [c.category_id for c in retained]

        self.assertIn(1, retained_ids)
        self.assertIn(2, retained_ids)

    def test_min_retained_candidates_constraint(self):
        # Even if over-suppressed, min_retained_candidates ensures pool budget is respected
        suppressor_high_min = AncestorSuppressor(
            dag=self.dag,
            child_confidence_threshold=0.35,
            min_retained_candidates=3,
        )

        candidates = [
            ScoredCandidate(category_id=0, name="Category:Architecture", score=0.90, source="stage1"),
            ScoredCandidate(category_id=1, name="Category:Gothic architecture", score=0.88, source="stage1"),
            ScoredCandidate(category_id=2, name="Category:Gothic cathedrals", score=0.85, source="stage1"),
        ]

        retained, decisions = suppressor_high_min.suppress(candidates)
        self.assertGreaterEqual(len(retained), 3)


if __name__ == "__main__":
    unittest.main()
