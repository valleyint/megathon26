"""Unit tests for OfflineReverseGeocoder."""

import unittest
from src.stage2.metadata.geocoder import OfflineReverseGeocoder, haversine_distance_km


class TestOfflineReverseGeocoder(unittest.TestCase):
    def setUp(self):
        self.geocoder = OfflineReverseGeocoder()

    def test_haversine_distance(self):
        # Distance between Eiffel Tower and Notre Dame is approx 4.2 km
        d = haversine_distance_km(48.8584, 2.2945, 48.8530, 2.3499)
        self.assertGreater(d, 3.5)
        self.assertLess(d, 5.0)

    def test_reverse_geocode_paris(self):
        # Eiffel Tower GPS: 48.8584 N, 2.2945 E
        loc = self.geocoder.reverse_geocode(48.8584, 2.2945)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.city, "Paris")
        self.assertEqual(loc.country, "France")
        self.assertEqual(loc.country_code, "FR")

    def test_reverse_geocode_london(self):
        # Big Ben GPS: 51.5007 N, -0.1246 W
        loc = self.geocoder.reverse_geocode(51.5007, -0.1246)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.city, "London")
        self.assertEqual(loc.country, "United Kingdom")

    def test_reverse_geocode_agra(self):
        # Taj Mahal GPS: 27.1751 N, 78.0421 E
        loc = self.geocoder.reverse_geocode(27.1751, 78.0421)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.city, "Agra")
        self.assertEqual(loc.country, "India")

    def test_max_distance_threshold(self):
        # Middle of the Atlantic Ocean
        loc = self.geocoder.reverse_geocode(0.0, 0.0, max_dist_km=50.0)
        self.assertIsNone(loc)

    def test_matches_location_name(self):
        loc = self.geocoder.reverse_geocode(48.8584, 2.2945)
        self.assertTrue(self.geocoder.matches_location_name(loc, "Paris"))
        self.assertTrue(self.geocoder.matches_location_name(loc, "France"))
        self.assertFalse(self.geocoder.matches_location_name(loc, "Berlin"))

    def test_add_custom_location(self):
        self.geocoder.add_location(
            city="Machu Picchu",
            country="Peru",
            lat=-13.1631,
            lon=-72.5450,
            radius_km=25.0,
        )
        loc = self.geocoder.reverse_geocode(-13.1631, -72.5450)
        self.assertIsNotNone(loc)
        self.assertEqual(loc.city, "Machu Picchu")
        self.assertEqual(loc.country, "Peru")


if __name__ == "__main__":
    unittest.main()
