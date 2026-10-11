"""Integration tests for the complete Stage 2 Pipeline."""

import unittest
from src.common.types import (
    Stage1CandidatePool,
    ScoredCandidate,
    ImageQuery,
)
from src.data.category_vocab import CategoryVocabulary
from src.stage2.graph.category_dag import CategoryDAG
from src.stage2.pipeline import Stage2Pipeline


class TestStage2Pipeline(unittest.TestCase):
    def setUp(self):
        # Build category vocabulary
        self.vocab = CategoryVocabulary()
        # Visual categories
        self.vocab.add_category("Architecture")  # 0
        self.vocab.add_category("Gothic architecture")  # 1
        self.vocab.add_category("Gothic cathedrals in France")  # 2
        self.vocab.add_category("Eiffel Tower")  # 3
        # Conflicting categories
        self.vocab.add_category("Bridges in London")  # 4 (Spatial conflict for Paris query)
        self.vocab.add_category("Demonstrations in Paris in 2018")  # 5 (Temporal conflict for 2023 query)
        # Event category
        self.vocab.add_category("Demonstrations and protests in Paris in 2023")  # 6

        # Build Category DAG
        self.dag = CategoryDAG(vocab=self.vocab)
        for cat_info in self.vocab.id_to_cat.values():
            self.dag.add_category(cat_info.category_id, cat_info.name)

        # 0 -> 1 -> 2 (Architecture -> Gothic architecture -> Gothic cathedrals in France)
        self.dag.add_edge(0, 1)
        self.dag.add_edge(1, 2)

        # Initialize pipeline
        self.pipeline = Stage2Pipeline(
            vocab=self.vocab,
            dag=self.dag,
            config={
                "filters": {
                    "temporal": {"year_tolerance": 1},
                    "spatial": {"strict_country_check": True},
                },
                "dag_pruning": {
                    "child_confidence_threshold": 0.35,
                },
                "stage2": {
                    "target_pool_size": 10,
                    "min_pool_size": 2,
                },
            },
        )

    def test_end_to_end_stage2_pipeline(self):
        # Mock Stage 1 Candidate Pool with 6 visual candidates
        s1_candidates = [
            ScoredCandidate(category_id=0, name="Category:Architecture", score=0.92, source="stage1"),
            ScoredCandidate(category_id=1, name="Category:Gothic architecture", score=0.88, source="stage1"),
            ScoredCandidate(category_id=2, name="Category:Gothic cathedrals in France", score=0.85, source="stage1"),
            ScoredCandidate(category_id=3, name="Category:Eiffel Tower", score=0.70, source="stage1"),
            ScoredCandidate(category_id=4, name="Category:Bridges in London", score=0.65, source="stage1"),
            ScoredCandidate(category_id=5, name="Category:Demonstrations in Paris in 2018", score=0.60, source="stage1"),
        ]

        stage1_pool = Stage1CandidatePool(
            query_id="paris_cathedral_query",
            embedding=None,
            candidates=s1_candidates,
            latency_ms={"vision_encoding": 18.5, "aggregation": 2.1},
        )

        query = ImageQuery(
            query_id="paris_cathedral_query",
            title="File:Notre_Dame_de_Paris_Protests_2023.jpg",
            description="{{Information|description=Protests near Gothic cathedrals in Paris during pension reforms}}",
            timestamp="2023:03:23 15:30:00",
            latitude=48.8530,
            longitude=2.3499,  # Paris GPS
        )

        stage2_pool = self.pipeline.process(stage1_pool, query)

        # 1. Output structure checks
        self.assertEqual(stage2_pool.query_id, "paris_cathedral_query")
        self.assertIsNotNone(stage2_pool.parsed_metadata)
        self.assertEqual(stage2_pool.parsed_metadata.capture_year, 2023)
        self.assertEqual(stage2_pool.parsed_metadata.location.city, "Paris")
        self.assertEqual(stage2_pool.parsed_metadata.location.country, "France")

        retained_ids = stage2_pool.category_ids

        # 2. Conflict pruning verification:
        # London should be pruned due to spatial conflict with Paris
        self.assertNotIn(4, retained_ids)

        # 2018 category should be pruned due to temporal conflict with 2023
        self.assertNotIn(5, retained_ids)

        # 3. DAG ancestor suppression verification:
        # Category 2 (Gothic cathedrals in France) is specific and confident (0.85 >= 0.35)
        # Broad ancestors 0 (Architecture) and 1 (Gothic architecture) must be suppressed!
        self.assertIn(2, retained_ids)
        self.assertNotIn(0, retained_ids)
        self.assertNotIn(1, retained_ids)

        # 4. Event discovery verification:
        # Event "Demonstrations and protests in Paris in 2023" should be proactively discovered & present
        self.assertIn(6, retained_ids)

        # 5. Audit trail & latency check
        self.assertGreater(len(stage2_pool.pruned_decisions), 0)
        self.assertIn("stage2_metadata_preprocessing", stage2_pool.latency_ms)
        self.assertIn("stage2_dag_ancestor_pruning", stage2_pool.latency_ms)
        self.assertIn("stage2_event_discovery", stage2_pool.latency_ms)


if __name__ == "__main__":
    unittest.main()
