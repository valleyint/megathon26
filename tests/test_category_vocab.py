"""Unit tests for CategoryVocabulary."""

import sys
import unittest
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.category_vocab import CategoryVocabulary


class TestCategoryVocabulary(unittest.TestCase):
    def setUp(self):
        self.vocab = CategoryVocabulary()

    def test_add_and_get_category(self):
        cat_id1 = self.vocab.add_category("Gothic architecture")
        cat_id2 = self.vocab.add_category("Category:Cathedrals in France")

        self.assertEqual(cat_id1, 0)
        self.assertEqual(cat_id2, 1)

        self.assertEqual(self.vocab.get_name(cat_id1), "Category:Gothic architecture")
        self.assertEqual(self.vocab.get_name(cat_id2), "Category:Cathedrals in France")

        self.assertEqual(self.vocab.get_id("Gothic architecture"), cat_id1)
        self.assertEqual(self.vocab.get_id("Category:Gothic architecture"), cat_id1)

    def test_duplicate_category(self):
        id1 = self.vocab.add_category("Eiffel Tower", frequency=5)
        id2 = self.vocab.add_category("Category:Eiffel Tower", frequency=3)

        self.assertEqual(id1, id2)
        self.assertEqual(len(self.vocab), 1)
        info = self.vocab.get_info(id1)
        self.assertEqual(info.frequency, 8)

    def test_save_and_load(self):
        self.vocab.add_category("Cat A", frequency=10)
        self.vocab.add_category("Cat B", frequency=20)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "vocab.json"
            self.vocab.save(file_path)

            loaded_vocab = CategoryVocabulary.load(file_path)
            self.assertEqual(len(loaded_vocab), 2)
            self.assertEqual(loaded_vocab.get_name(0), "Category:Cat A")
            self.assertEqual(loaded_vocab.get_name(1), "Category:Cat B")
            self.assertEqual(loaded_vocab.get_info(1).frequency, 20)


if __name__ == "__main__":
    unittest.main()
