"""Temporal consistency filtering based on EXIF capture dates and category years."""

from __future__ import annotations
import re
from typing import List, Tuple, Optional

from src.common.types import ScoredCandidate, PruningDecision, ParsedMetadata


class TemporalConflictFilter:
    """Prunes categories with explicit year tags violating the image's EXIF capture year."""

    def __init__(self, year_tolerance: int = 1):
        self.year_tolerance = year_tolerance
        self._year_re = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
        # Century and era exclusions (e.g., "19th-century", "20th century")
        self._century_re = re.compile(r"\b(?:1[6-9]|20)th[-\s]century\b", re.IGNORECASE)

    def extract_category_year(self, category_name: str) -> Optional[int]:
        """Extracts event/capture year from category name, skipping broad historical centuries."""
        clean = category_name.replace("Category:", "")

        # Skip centuries e.g. "19th-century architecture"
        if self._century_re.search(clean):
            return None

        matches = self._year_re.findall(clean)
        if matches:
            try:
                # Return the last matched year (often the primary event year in title)
                return int(matches[-1])
            except ValueError:
                return None
        return None

    def filter(
        self,
        candidates: List[ScoredCandidate],
        parsed_metadata: Optional[ParsedMetadata],
    ) -> Tuple[List[ScoredCandidate], List[PruningDecision]]:
        """Filters candidate pool against metadata capture year."""
        if not parsed_metadata or parsed_metadata.capture_year is None:
            # No verified capture year; cannot reject temporally
            return candidates, []

        capture_year = parsed_metadata.capture_year
        retained: List[ScoredCandidate] = []
        decisions: List[PruningDecision] = []

        for cand in candidates:
            cand_year = self.extract_category_year(cand.name)
            if cand_year is not None:
                diff = abs(cand_year - capture_year)
                if diff > self.year_tolerance:
                    # Violates chronological consistency
                    decisions.append(
                        PruningDecision(
                            category_id=cand.category_id,
                            name=cand.name,
                            status="temporal_conflict",
                            score=cand.score,
                            reason=(
                                f"Explicit category year {cand_year} contradicts "
                                f"EXIF capture year {capture_year} (diff={diff} > {self.year_tolerance})"
                            ),
                            metadata={"cand_year": cand_year, "capture_year": capture_year},
                        )
                    )
                    continue

            retained.append(cand)

        return retained, decisions
