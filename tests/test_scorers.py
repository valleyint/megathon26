"""Unit tests for Stage 1 Scorers (KNNScorer and PrototypeScorer)."""

import sys
import unittest
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import numpy as np
except ImportError:
    np = None

from src.data.category_vocab import CategoryVocabulary
from src.stage1.scorers.prototype_scorer import PrototypeScorer
from src.stage1.scorers.knn_scorer import KNNScorer


@unittest.skipIf(np is None, "numpy not installed in current environment")
class TestScorers(unittest.TestCase):
    def setUp(self):
        self.vocab = CategoryVocabulary()
        self.vocab.add_category("Gothic architecture")
        self.vocab.add_category("Romanesque churches")
        self.vocab.add_category("High-speed rail")

    def test_prototype_scorer(self):
        # 3 categories with 4-dimensional vectors
        prototypes = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Gothic
            [0.0, 1.0, 0.0, 0.0],  # Romanesque
            [0.0, 0.0, 1.0, 0.0],  # Rail
        ], dtype=np.float32)

        scorer = PrototypeScorer(prototypes=prototypes, vocab=self.vocab)

        # Query vector aligned with Gothic
        query_z = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        query_z /= np.linalg.norm(query_z)

        results = scorer.score(query_z, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].category_id, 0)
        self.assertEqual(results[0].name, "Category:Gothic architecture")
        self.assertGreater(results[0].score, results[1].score)

        # Dynamic addition (Continual Learning)
        new_vec = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        new_cat_id = scorer.add_category("Impressionist art", new_vec)
        self.assertEqual(new_cat_id, 3)
        self.assertEqual(len(scorer.prototypes), 4)

        # Query aligned with new category
        query_new = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        res_new = scorer.score(query_new, top_k=1)
        self.assertEqual(res_new[0].category_id, new_cat_id)

    def test_knn_scorer_consensus(self):
        scorer = KNNScorer(temperature=0.1, top_k_neighbors=3, vocab=self.vocab)

        # 4 historical images
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        image_ids = ["img_1", "img_2", "img_3", "img_4"]
        labels = [
            [0],     # img_1 has Gothic
            [0, 1],  # img_2 has Gothic & Romanesque
            [1],     # img_3 has Romanesque
            [2],     # img_4 has Rail
        ]

        scorer.add_historical_images(vectors, image_ids, labels)

        # Query aligned with img_1 and img_2
        query_z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = scorer.score(query_z, top_k=2)

        self.assertGreaterEqual(len(results), 1)
        # Gothic should have highest consensus
        self.assertEqual(results[0].category_id, 0)
        self.assertEqual(results[0].name, "Category:Gothic architecture")
        self.assertIn("neighbor_exemplars", results[0].metadata)
        self.assertIn("img_1", results[0].metadata["neighbor_exemplars"])

    def test_save_and_load_scorers(self):
        prototypes = np.eye(3, dtype=np.float32)
        proto_scorer = PrototypeScorer(prototypes=prototypes, vocab=self.vocab)

        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_file = Path(tmp_dir) / "prototypes.npy"
            vocab_file = Path(tmp_dir) / "vocab.json"
            proto_scorer.save(proto_file, vocab_file)

            loaded_proto = PrototypeScorer(prototypes_path=proto_file, vocab_path=vocab_file)
            self.assertEqual(len(loaded_proto.prototypes), 3)
            self.assertEqual(len(loaded_proto.vocab), 3)


if __name__ == "__main__":
    unittest.main()
