"""Proactive Commons Event Hypothesis Generation."""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from src.common.types import ParsedMetadata, ScoredCandidate


class EventHypothesisGenerator:
    """Synthesizes Wikimedia Commons event category hypotheses from visual anchors and metadata."""

    def __init__(self, use_llm: bool = False, llm_client: Optional[Any] = None):
        self.use_llm = use_llm
        self.llm_client = llm_client

    def generate_hypotheses(
        self,
        visual_anchors: List[ScoredCandidate],
        parsed_metadata: ParsedMetadata,
        max_hypotheses: int = 25,
    ) -> List[str]:
        """Synthesizes structured event category hypotheses following Commons naming conventions."""
        hypotheses: List[str] = []
        seen = set()

        def add_hypothesis(cand: str):
            clean = cand.strip()
            if not clean.startswith("Category:"):
                clean = f"Category:{clean}"
            if clean.lower() not in seen:
                seen.add(clean.lower())
                hypotheses.append(clean)

        year = parsed_metadata.capture_year
        loc = parsed_metadata.location
        city = loc.city if loc and loc.city else ""
        country = loc.country if loc and loc.country else ""
        entities = parsed_metadata.extracted_entities

        # Extract anchor topics by stripping "Category:" prefix
        anchor_topics = []
        for cand in visual_anchors[:10]:
            name = cand.name.replace("Category:", "").strip()
            # Exclude very broad or metadata-like topics
            if name and len(name) > 2:
                anchor_topics.append(name)

        # 1. Topic in City in Year: "Category:[Topic] in [City] in [YYYY]"
        if year and city:
            for topic in anchor_topics[:5]:
                add_hypothesis(f"{topic} in {city} in {year}")
                add_hypothesis(f"{topic} in {city}")

        # 2. Topic in Country in Year: "Category:[Topic] in [Country] in [YYYY]"
        if year and country:
            for topic in anchor_topics[:5]:
                add_hypothesis(f"{topic} in {country} in {year}")
                add_hypothesis(f"{topic} in {country}")

        # 3. Year in Location: "Category:[YYYY] in [City/Country]"
        if year and city:
            add_hypothesis(f"{year} in {city}")
        if year and country:
            add_hypothesis(f"{year} in {country}")

        # 4. Entities from Title / Description combined with Year
        # e.g., "Tour de France 2023", "2024 Summer Olympics", "Bastille Day 2023"
        for entity in entities[:5]:
            if year:
                add_hypothesis(f"{year} {entity}")
                add_hypothesis(f"{entity} {year}")
                if city:
                    add_hypothesis(f"{entity} in {city}")
                if country:
                    add_hypothesis(f"{entity} in {country}")
            else:
                add_hypothesis(f"{entity}")

        # 5. Clean Title as Direct Event Hypothesis
        if parsed_metadata.title_clean:
            title_clean = parsed_metadata.title_clean
            add_hypothesis(title_clean)
            if year and str(year) not in title_clean:
                add_hypothesis(f"{title_clean} in {year}")

        # 6. LLM Synthesis (if configured and enabled)
        if self.use_llm and self.llm_client:
            llm_results = self._call_llm_synthesizer(visual_anchors, parsed_metadata)
            for h in llm_results:
                add_hypothesis(h)

        return hypotheses[:max_hypotheses]

    def _call_llm_synthesizer(
        self,
        visual_anchors: List[ScoredCandidate],
        parsed_metadata: ParsedMetadata,
    ) -> List[str]:
        """Calls optional instruction LLM for creative event taxonomy synthesis."""
        try:
            prompt = (
                f"Given the visual concepts: {[c.name for c in visual_anchors[:5]]}, "
                f"year: {parsed_metadata.capture_year}, location: {parsed_metadata.location}, "
                f"title: '{parsed_metadata.title_clean}', "
                f"generate 5 canonical Wikimedia Commons event category names."
            )
            if hasattr(self.llm_client, "generate"):
                response = self.llm_client.generate(prompt)
                return [line.strip("- ") for line in response.splitlines() if line.strip()]
        except Exception:
            pass
        return []
