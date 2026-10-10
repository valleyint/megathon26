"""Integration test for the complete Stage 1 Pipeline."""

import sys
import unittest
from pathlib import Path
import tempfile

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import numpy as np
except ImportError:
    np = None

from src.data.category_vocab import CategoryVocabulary
from src.encoders.siglip_encoder import SigLIPVisionEncoder
from src.stage1.scorers.prototype_scorer import PrototypeScorer
from src.stage1.scorers.knn_scorer import KNNScorer
from src.stage1.aggregator import Stage1Aggregator
from src.stage1.pipeline import Stage1Pipeline


class TestStage1Pipeline(unittest.TestCase):
    def setUp(self):
        self.vocab = CategoryVocabulary()
        self.vocab.add_category("Gothic architecture")
        self.vocab.add_category("Romanesque churches")
        self.vocab.add_category("High-speed rail")

        # Mock vision encoder producing 768-d unit vectors
        self.vision_encoder = SigLIPVisionEncoder(mock_dim=768)

        # Mock prototypes matrix: shape (3, 768)
        prototypes = np.random.randn(3, 768).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=-1, keepdims=True)

        self.proto_scorer = PrototypeScorer(prototypes=prototypes, vocab=self.vocab)
        self.knn_scorer = KNNScorer(temperature=0.07, top_k_neighbors=5, vocab=self.vocab)

        # Add mock historical images to KNN
        hist_vectors = np.random.randn(10, 768).astype(np.float32)
        hist_vectors /= np.linalg.norm(hist_vectors, axis=-1, keepdims=True)
        img_ids = [f"hist_{i}" for i in range(10)]
        labels = [[i % 3] for i in range(10)]
        self.knn_scorer.add_historical_images(hist_vectors, img_ids, labels)

        self.aggregator = Stage1Aggregator(pool_size=3)
        self.pipeline = Stage1Pipeline(
            vision_encoder=self.vision_encoder,
            scorers={"prototype": self.proto_scorer, "knn": self.knn_scorer},
            aggregator=self.aggregator,
        )

    def test_pipeline_prediction(self):
        # Create a dummy image array (224x224 RGB)
        dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

        pool = self.pipeline.predict(dummy_image, query_id="test_query_001")

        self.assertEqual(pool.query_id, "test_query_001")
        self.assertEqual(len(pool.candidates), 3)
        self.assertEqual(pool.embedding.shape, (768,))

        # Verify latency tracking
        self.assertIn("vision_encoding", pool.latency_ms)
        self.assertIn("scorer_prototype", pool.latency_ms)
        self.assertIn("scorer_knn", pool.latency_ms)
        self.assertIn("aggregation", pool.latency_ms)
        self.assertIn("total_ms", pool.latency_ms)
        self.assertGreater(pool.latency_ms["total_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
