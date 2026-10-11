"""Unit tests for WikitextMetadataCleaner."""

import unittest
from src.stage2.metadata.cleaner import WikitextMetadataCleaner


class TestWikitextMetadataCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = WikitextMetadataCleaner()

    def test_clean_wikitext_templates(self):
        raw = "This is a photo {{Information |description=Eiffel Tower {{lang|fr|Tour Eiffel}} |date=2023 }} of Paris."
        clean = self.cleaner.clean_text(raw)
        self.assertNotIn("{{", clean)
        self.assertNotIn("}}", clean)
        self.assertIn("Eiffel Tower", clean)
        self.assertIn("Paris", clean)

    def test_clean_wikilinks_and_comments(self):
        raw = "View of [[Eiffel Tower|The Iron Lady]] <!-- comment --> taken in [[Paris]]."
        clean = self.cleaner.clean_text(raw)
        self.assertEqual(clean, "View of The Iron Lady taken in Paris.")

    def test_clean_title(self):
        title = "File:Eiffel_Tower_from_the_Trocadéro_2023.jpg"
        clean = self.cleaner.clean_title(title)
        self.assertEqual(clean, "Eiffel Tower from the Trocadéro 2023")

    def test_extract_year_from_exif(self):
        ts1 = "2023:07:14 18:45:00"
        self.assertEqual(self.cleaner.extract_year(ts1), 2023)

        ts2 = "2019-10-05T12:00:00Z"
        self.assertEqual(self.cleaner.extract_year(ts2), 2019)

    def test_extract_year_from_text_fallback(self):
        text = "Protest during the French pension reform demonstrations in Paris 2023."
        year = self.cleaner.extract_year(None, text_fallback=text)
        self.assertEqual(year, 2023)

    def test_extract_named_entities(self):
        text = "Photograph of Eiffel Tower during Bastille Day celebrations in Paris."
        entities = self.cleaner.extract_named_entities(text)
        self.assertIn("Eiffel Tower", entities)
        self.assertIn("Bastille Day", entities)
        self.assertIn("Paris", entities)

    def test_full_parse(self):
        parsed = self.cleaner.parse(
            title="File:Notre_Dame_de_Paris_2018.jpg",
            description="{{Information|description=Cathedral Notre Dame}}",
            timestamp="2018:04:15 10:00:00",
            latitude=48.8530,
            longitude=2.3499,
        )
        self.assertEqual(parsed.title_clean, "Notre Dame de Paris 2018")
        self.assertEqual(parsed.capture_year, 2018)
        self.assertIsNotNone(parsed.location)
        self.assertAlmostEqual(parsed.location.latitude, 48.8530)
        self.assertAlmostEqual(parsed.location.longitude, 2.3499)


if __name__ == "__main__":
    unittest.main()
