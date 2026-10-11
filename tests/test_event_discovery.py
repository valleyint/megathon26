"""Unit tests for EventHypothesisGenerator and EventIndexRetriever."""

import unittest
from src.common.types import ScoredCandidate, ParsedMetadata, GeoLocation
from src.data.category_vocab import CategoryVocabulary
from src.stage2.events.hypothesis_generator import EventHypothesisGenerator
from src.stage2.events.event_retriever import EventIndexRetriever, BM25Index


class TestEventDiscovery(unittest.TestCase):
    def setUp(self):
        self.vocab = CategoryVocabulary()
        self.vocab.add_category("Demonstrations and protests in Paris in 2023")
        self.vocab.add_category("2024 Summer Olympics")
        self.vocab.add_category("Eiffel Tower in 2023")
        self.vocab.add_category("Bridges in London")
        self.vocab.add_category("Gothic architecture in France")

        self.generator = EventHypothesisGenerator()
        self.retriever = EventIndexRetriever(vocab=self.vocab)

    def test_hypothesis_generation(self):
        visual_cands = [
            ScoredCandidate(category_id=0, name="Category:Demonstrations and protests", score=0.85, source="stage1"),
            ScoredCandidate(category_id=1, name="Category:Eiffel Tower", score=0.75, source="stage1"),
        ]
        parsed_meta = ParsedMetadata(
            title_clean="Retirement reform protests near Eiffel Tower",
            capture_year=2023,
            location=GeoLocation(latitude=48.8584, longitude=2.2945, city="Paris", country="France"),
            extracted_entities=["Retirement", "Eiffel Tower"],
        )

        hypotheses = self.generator.generate_hypotheses(visual_cands, parsed_meta)

        self.assertIn("Category:Demonstrations and protests in Paris in 2023", hypotheses)
        self.assertIn("Category:Eiffel Tower in Paris in 2023", hypotheses)
        self.assertIn("Category:2023 in Paris", hypotheses)
        self.assertIn("Category:2023 in France", hypotheses)

    def test_bm25_index(self):
        corpus = [(info.category_id, info.name) for info in self.vocab.id_to_cat.values()]
        bm25 = BM25Index(corpus)
        hits = bm25.query("protests Paris 2023", top_k=2)

        self.assertGreater(len(hits), 0)
        top_doc_idx, top_score = hits[0]
        self.assertEqual(bm25.doc_names[top_doc_idx], "Category:Demonstrations and protests in Paris in 2023")

    def test_event_retrieval_and_snap(self):
        hypotheses = [
            "Category:Demonstrations and protests in Paris in 2023",
            "Category:Unknown non-existent event 2023",
        ]
        results = self.retriever.retrieve_and_snap(hypotheses, top_k=5)

        # Must snap to verified categories from vocabulary only
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].name, "Category:Demonstrations and protests in Paris in 2023")
        self.assertEqual(results[0].source, "event_retrieval")

        # Unknown hallucinated category is never returned as an unverified node
        returned_names = [r.name for r in results]
        self.assertNotIn("Category:Unknown non-existent event 2023", returned_names)


if __name__ == "__main__":
    unittest.main()
