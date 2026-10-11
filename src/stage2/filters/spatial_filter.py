"""Spatial consistency filtering based on GPS geocoding and category locations."""

from __future__ import annotations
import re
from typing import List, Tuple, Optional, Set, Dict

from src.common.types import ScoredCandidate, PruningDecision, ParsedMetadata, GeoLocation
from src.stage2.metadata.geocoder import OfflineReverseGeocoder, DEFAULT_WORLD_CITIES


class SpatialConflictFilter:
    """Prunes categories with explicit location references contradicting GPS geotags."""

    def __init__(
        self,
        geocoder: Optional[OfflineReverseGeocoder] = None,
        strict_country_check: bool = True,
    ):
        self.geocoder = geocoder or OfflineReverseGeocoder()
        self.strict_country_check = strict_country_check

        # Build lookup table of known city and country names
        self.city_to_country: Dict[str, str] = {}
        self.known_locations: Set[str] = set()

        for entry in self.geocoder.gazetteer:
            city_l = entry["city"].lower()
            country_l = entry["country"].lower()
            self.city_to_country[city_l] = country_l
            self.known_locations.add(city_l)
            self.known_locations.add(country_l)

    def extract_location_mentions(self, category_name: str) -> List[str]:
        """Finds known cities or countries referenced in category name."""
        clean = category_name.replace("Category:", "").lower()
        found = []
        for loc in self.known_locations:
            pattern = rf"\b{re.escape(loc)}\b"
            if re.search(pattern, clean):
                found.append(loc)
        return found

    def filter(
        self,
        candidates: List[ScoredCandidate],
        parsed_metadata: Optional[ParsedMetadata],
    ) -> Tuple[List[ScoredCandidate], List[PruningDecision]]:
        """Filters candidates against resolved GPS GeoLocation."""
        if not parsed_metadata or not parsed_metadata.location:
            # No GPS location available; cannot filter spatially
            return candidates, []

        geo = parsed_metadata.location
        geo_city = geo.city.lower() if geo.city else ""
        geo_country = geo.country.lower() if geo.country else ""

        retained: List[ScoredCandidate] = []
        decisions: List[PruningDecision] = []

        for cand in candidates:
            loc_mentions = self.extract_location_mentions(cand.name)
            if loc_mentions:
                is_contradiction = False
                conflicting_loc = ""

                for loc in loc_mentions:
                    # If mentioned loc is a city
                    if loc in self.city_to_country:
                        cand_country = self.city_to_country[loc]
                        # If image GPS city is known and doesn't match mentioned city
                        if geo_city and loc != geo_city:
                            is_contradiction = True
                            conflicting_loc = loc
                            break
                        # If image country doesn't match city's country
                        if geo_country and cand_country != geo_country:
                            is_contradiction = True
                            conflicting_loc = loc
                            break

                    # If mentioned loc is a country
                    elif geo_country and loc != geo_country:
                        is_contradiction = True
                        conflicting_loc = loc
                        break

                if is_contradiction:
                    decisions.append(
                        PruningDecision(
                            category_id=cand.category_id,
                            name=cand.name,
                            status="spatial_conflict",
                            score=cand.score,
                            reason=(
                                f"Category references location '{conflicting_loc.title()}' which contradicts "
                                f"GPS geotag resolved to '{geo.city or 'Unknown'}, {geo.country or 'Unknown'}'"
                            ),
                            metadata={"conflicting_location": conflicting_loc, "gps_location": geo.to_dict()},
                        )
                    )
                    continue

            retained.append(cand)

        return retained, decisions
