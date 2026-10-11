"""Unit tests for TemporalConflictFilter and SpatialConflictFilter."""

import unittest
from src.common.types import ScoredCandidate, ParsedMetadata, GeoLocation
from src.stage2.filters.temporal_filter import TemporalConflictFilter
from src.stage2.filters.spatial_filter import SpatialConflictFilter


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.temporal_filter = TemporalConflictFilter(year_tolerance=1)
        self.spatial_filter = SpatialConflictFilter()

    def test_temporal_filter_prunes_conflicts(self):
        # Capture year is 2023
        parsed_meta = ParsedMetadata(capture_year=2023)

        candidates = [
            ScoredCandidate(category_id=0, name="Category:Protests in Paris in 2023", score=0.9, source="stage1"),
            ScoredCandidate(category_id=1, name="Category:Events in 2022 in France", score=0.8, source="stage1"),  # Tol=1, keep
            ScoredCandidate(category_id=2, name="Category:Demonstrations in Paris in 2018", score=0.7, source="stage1"),  # 2018 != 2023, drop!
            ScoredCandidate(category_id=3, name="Category:Eiffel Tower", score=0.85, source="stage1"),  # No year, keep!
            ScoredCandidate(category_id=4, name="Category:19th-century architecture in Paris", score=0.75, source="stage1"),  # Century, keep!
        ]

        retained, decisions = self.temporal_filter.filter(candidates, parsed_meta)
        retained_ids = [c.category_id for c in retained]

        self.assertIn(0, retained_ids)
        self.assertIn(1, retained_ids)
        self.assertNotIn(2, retained_ids)
        self.assertIn(3, retained_ids)
        self.assertIn(4, retained_ids)

        # Check pruning decisions
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].category_id, 2)
        self.assertEqual(decisions[0].status, "temporal_conflict")

    def test_spatial_filter_prunes_conflicts(self):
        # GPS is resolved to Paris, France
        parsed_meta = ParsedMetadata(
            location=GeoLocation(latitude=48.8584, longitude=2.2945, city="Paris", country="France")
        )

        candidates = [
            ScoredCandidate(category_id=0, name="Category:Eiffel Tower in Paris", score=0.9, source="stage1"),
            ScoredCandidate(category_id=1, name="Category:Bridges in London", score=0.8, source="stage1"),  # London contradicts Paris, drop!
            ScoredCandidate(category_id=2, name="Category:Colosseum in Rome", score=0.7, source="stage1"),  # Rome contradicts Paris, drop!
            ScoredCandidate(category_id=3, name="Category:Gothic architecture in France", score=0.85, source="stage1"),  # France matches, keep!
            ScoredCandidate(category_id=4, name="Category:Public parks", score=0.6, source="stage1"),  # No location, keep!
        ]

        retained, decisions = self.spatial_filter.filter(candidates, parsed_meta)
        retained_ids = [c.category_id for c in retained]

        self.assertIn(0, retained_ids)
        self.assertNotIn(1, retained_ids)
        self.assertNotIn(2, retained_ids)
        self.assertIn(3, retained_ids)
        self.assertIn(4, retained_ids)

        # Check decisions
        pruned_statuses = [d.status for d in decisions]
        self.assertEqual(pruned_statuses, ["spatial_conflict", "spatial_conflict"])


if __name__ == "__main__":
    unittest.main()
