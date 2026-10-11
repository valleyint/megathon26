"""Offline reverse geocoder using spatial distance lookups and geographic gazetteer."""

from __future__ import annotations
import math
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.common.types import GeoLocation


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0  # Earth's mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


DEFAULT_WORLD_CITIES = [
    # Europe
    {"city": "Paris", "state": "Île-de-France", "country": "France", "code": "FR", "lat": 48.8566, "lon": 2.3522, "radius": 40.0},
    {"city": "London", "state": "England", "country": "United Kingdom", "code": "GB", "lat": 51.5074, "lon": -0.1278, "radius": 50.0},
    {"city": "Rome", "state": "Lazio", "country": "Italy", "code": "IT", "lat": 41.9028, "lon": 12.4964, "radius": 35.0},
    {"city": "Berlin", "state": "Berlin", "country": "Germany", "code": "DE", "lat": 52.5200, "lon": 13.4050, "radius": 40.0},
    {"city": "Madrid", "state": "Community of Madrid", "country": "Spain", "code": "ES", "lat": 40.4168, "lon": -3.7038, "radius": 40.0},
    {"city": "Amsterdam", "state": "North Holland", "country": "Netherlands", "code": "NL", "lat": 52.3676, "lon": 4.9041, "radius": 30.0},
    {"city": "Vienna", "state": "Vienna", "country": "Austria", "code": "AT", "lat": 48.2082, "lon": 16.3738, "radius": 30.0},
    {"city": "Athens", "state": "Attica", "country": "Greece", "code": "GR", "lat": 37.9838, "lon": 23.7275, "radius": 30.0},
    {"city": "Venice", "state": "Veneto", "country": "Italy", "code": "IT", "lat": 45.4408, "lon": 12.3155, "radius": 25.0},
    {"city": "Barcelona", "state": "Catalonia", "country": "Spain", "code": "ES", "lat": 41.3851, "lon": 2.1734, "radius": 35.0},

    # Asia & Middle East
    {"city": "Agra", "state": "Uttar Pradesh", "country": "India", "code": "IN", "lat": 27.1751, "lon": 78.0421, "radius": 30.0},
    {"city": "New Delhi", "state": "Delhi", "country": "India", "code": "IN", "lat": 28.6139, "lon": 77.2090, "radius": 40.0},
    {"city": "Tokyo", "state": "Tokyo", "country": "Japan", "code": "JP", "lat": 35.6762, "lon": 139.6503, "radius": 50.0},
    {"city": "Kyoto", "state": "Kyoto", "country": "Japan", "code": "JP", "lat": 35.0116, "lon": 135.7681, "radius": 30.0},
    {"city": "Beijing", "state": "Beijing", "country": "China", "code": "CN", "lat": 39.9042, "lon": 116.4074, "radius": 50.0},
    {"city": "Singapore", "state": "Singapore", "country": "Singapore", "code": "SG", "lat": 1.3521, "lon": 103.8198, "radius": 30.0},
    {"city": "Dubai", "state": "Dubai", "country": "United Arab Emirates", "code": "AE", "lat": 25.2048, "lon": 55.2708, "radius": 40.0},
    {"city": "Cairo", "state": "Cairo", "country": "Egypt", "code": "EG", "lat": 30.0444, "lon": 31.2357, "radius": 40.0},
    {"city": "Jerusalem", "state": "Jerusalem", "country": "Israel", "code": "IL", "lat": 31.7683, "lon": 35.2137, "radius": 25.0},

    # Americas
    {"city": "New York", "state": "New York", "country": "United States", "code": "US", "lat": 40.7128, "lon": -74.0060, "radius": 50.0},
    {"city": "Washington, D.C.", "state": "District of Columbia", "country": "United States", "code": "US", "lat": 38.9072, "lon": -77.0369, "radius": 35.0},
    {"city": "San Francisco", "state": "California", "country": "United States", "code": "US", "lat": 37.7749, "lon": -122.4194, "radius": 35.0},
    {"city": "Rio de Janeiro", "state": "Rio de Janeiro", "country": "Brazil", "code": "BR", "lat": -22.9068, "lon": -43.1729, "radius": 40.0},
    {"city": "Toronto", "state": "Ontario", "country": "Canada", "code": "CA", "lat": 43.6532, "lon": -79.3832, "radius": 40.0},

    # Oceania & Africa
    {"city": "Sydney", "state": "New South Wales", "country": "Australia", "code": "AU", "lat": -33.8688, "lon": 151.2093, "radius": 40.0},
    {"city": "Cape Town", "state": "Western Cape", "country": "South Africa", "code": "ZA", "lat": -33.9249, "lon": 18.4241, "radius": 35.0},
]


class OfflineReverseGeocoder:
    """Fast, offline reverse geocoder resolving GPS coordinates to administrative entities in < 2ms."""

    def __init__(self, custom_gazetteer: Optional[List[Dict[str, Any]]] = None):
        self.gazetteer: List[Dict[str, Any]] = custom_gazetteer if custom_gazetteer is not None else DEFAULT_WORLD_CITIES
        self._kd_tree = None
        self._coords = None
        self._init_spatial_index()

    def _init_spatial_index(self) -> None:
        """Initializes spatial acceleration structure if scipy is available, else uses pure Python."""
        try:
            import numpy as np
            from scipy.spatial import KDTree

            # Convert (lat, lon) to 3D Cartesian coordinates on unit sphere for exact Euclidean k-NN
            coords_3d = []
            for item in self.gazetteer:
                lat_rad = math.radians(item["lat"])
                lon_rad = math.radians(item["lon"])
                x = math.cos(lat_rad) * math.cos(lon_rad)
                y = math.cos(lat_rad) * math.sin(lon_rad)
                z = math.sin(lat_rad)
                coords_3d.append([x, y, z])

            self._coords = np.array(coords_3d, dtype=np.float32)
            self._kd_tree = KDTree(self._coords)
        except (ImportError, Exception):
            self._kd_tree = None
            self._coords = None

    def add_location(
        self,
        city: str,
        country: str,
        lat: float,
        lon: float,
        state: Optional[str] = None,
        code: Optional[str] = None,
        radius_km: float = 30.0,
    ) -> None:
        """Dynamically appends a geographic location to the gazetteer."""
        entry = {
            "city": city,
            "state": state or city,
            "country": country,
            "code": code or "",
            "lat": float(lat),
            "lon": float(lon),
            "radius": float(radius_km),
        }
        self.gazetteer.append(entry)
        self._init_spatial_index()

    def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        max_dist_km: float = 150.0,
    ) -> Optional[GeoLocation]:
        """Resolves GPS coordinates to the closest administrative entity within max_dist_km."""
        if latitude is None or longitude is None:
            return None

        lat_f = float(latitude)
        lon_f = float(longitude)

        # Fast path via KDTree if available
        if self._kd_tree is not None:
            import numpy as np
            lat_rad = math.radians(lat_f)
            lon_rad = math.radians(lon_f)
            q_3d = np.array([
                math.cos(lat_rad) * math.cos(lon_rad),
                math.cos(lat_rad) * math.sin(lon_rad),
                math.sin(lat_rad)
            ], dtype=np.float32)

            # Query nearest neighbor
            dist_chord, idx = self._kd_tree.query(q_3d, k=1)
            # Great circle distance d = 2 * R * arcsin(chord / 2)
            dist_km = 2.0 * 6371.0 * math.asin(min(1.0, float(dist_chord) / 2.0))
            best_entry = self.gazetteer[idx]
        else:
            # Deterministic, pure-Python linear scan (takes < 0.1ms for 100-1000 entries)
            best_entry = None
            dist_km = float("inf")
            for entry in self.gazetteer:
                d = haversine_distance_km(lat_f, lon_f, entry["lat"], entry["lon"])
                if d < dist_km:
                    dist_km = d
                    best_entry = entry

        if best_entry is not None and dist_km <= max_dist_km:
            # Compute approximate bounding box (lat +/- delta, lon +/- delta)
            delta_lat = best_entry.get("radius", 30.0) / 111.0
            delta_lon = best_entry.get("radius", 30.0) / (111.0 * max(0.1, math.cos(math.radians(lat_f))))
            bbox = (
                best_entry["lat"] - delta_lat,
                best_entry["lon"] - delta_lon,
                best_entry["lat"] + delta_lat,
                best_entry["lon"] + delta_lon,
            )

            return GeoLocation(
                latitude=lat_f,
                longitude=lon_f,
                city=best_entry["city"],
                state=best_entry.get("state"),
                country=best_entry["country"],
                country_code=best_entry.get("code"),
                bounding_box=bbox,
            )

        return None

    def matches_location_name(
        self,
        geo: GeoLocation,
        target_name: str,
    ) -> bool:
        """Checks if a resolved GeoLocation matches a given city/state/country name."""
        if not geo or not target_name:
            return False

        target = target_name.strip().lower()
        candidates = [
            geo.city.lower() if geo.city else "",
            geo.state.lower() if geo.state else "",
            geo.country.lower() if geo.country else "",
            geo.country_code.lower() if geo.country_code else "",
        ]

        for cand in candidates:
            if cand and (target in cand or cand in target):
                return True
        return False
