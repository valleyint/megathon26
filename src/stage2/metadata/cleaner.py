"""Wikitext metadata cleaning, template stripping, and EXIF signal extraction."""

from __future__ import annotations
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.common.types import ParsedMetadata, GeoLocation


class WikitextMetadataCleaner:
    """Cleans raw Wikimedia metadata, strips boilerplate, and extracts temporal/spatial signals."""

    def __init__(self):
        # Compiled regexes for high-throughput string cleaning
        self._html_tag_re = re.compile(r"<[^>]+>", re.DOTALL)
        self._html_comment_re = re.compile(r"<!--.*?-->", re.DOTALL)
        self._template_re = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
        self._wikilink_re = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
        self._extlink_re = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]+))?\]")
        self._cat_link_re = re.compile(r"\[\[Category:([^\]]+)\]\]", re.IGNORECASE)
        self._whitespace_re = re.compile(r"\s+")
        self._year_re = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")

    def clean_text(self, text: Optional[str]) -> str:
        """Strips wiki markup, HTML, templates, and external links from text."""
        if not text:
            return ""

        s = str(text)

        # 1. Strip HTML comments
        s = self._html_comment_re.sub(" ", s)

        # 2. Strip Category tags
        s = self._cat_link_re.sub(" ", s)

        # 2.5 Extract content from metadata templates like {{Information|description=...}}
        info_match = re.search(r"\{\{(?:Information|Artwork|Photograph)[^|}]*\|(.*?)\}\}", s, re.DOTALL | re.IGNORECASE)
        if info_match:
            params_str = info_match.group(1)
            # Find description parameter
            desc_match = re.search(r"(?:description|desc|caption)\s*=\s*([^|\}]+)", params_str, re.IGNORECASE)
            if desc_match:
                extracted_content = desc_match.group(1).strip()
                s = s[:info_match.start()] + " " + extracted_content + " " + s[info_match.end():]

        # 3. Iteratively remove nested Wiki templates {{...}}
        for _ in range(5):
            new_s = self._template_re.sub(" ", s)
            if new_s == s:
                break
            s = new_s

        # 4. Resolve wiki links [[Target|Anchor]] -> Anchor
        s = self._wikilink_re.sub(r"\1", s)

        # 5. Resolve external links [http... Label] -> Label
        s = self._extlink_re.sub(lambda m: m.group(1) or "", s)

        # 6. Strip HTML tags
        s = self._html_tag_re.sub(" ", s)

        # 7. Normalize whitespace
        s = self._whitespace_re.sub(" ", s).strip()
        return s

    def clean_title(self, title: Optional[str]) -> str:
        """Cleans Wikimedia Commons file titles: removes File: prefix, extensions, and underscores."""
        if not title:
            return ""

        t = str(title).strip()
        # Remove File: or Image: prefix
        t = re.sub(r"^(?:File|Image):", "", t, flags=re.IGNORECASE)

        # Remove common image extensions
        t = re.sub(r"\.(?:jpe?g|png|gif|svg|tif?f|webp|bmp)$", "", t, flags=re.IGNORECASE)

        # Replace underscores with spaces
        t = t.replace("_", " ")

        # Strip wiki formatting if any
        t = self.clean_text(t)
        return t

    def extract_year(self, timestamp: Optional[str], text_fallback: str = "") -> Optional[int]:
        """Extracts the 4-digit capture year from EXIF/ISO timestamp, falling back to text."""
        if timestamp:
            ts_str = str(timestamp).strip()
            # 1. Match common EXIF / ISO formats like "YYYY:MM:DD HH:MM:SS" or "YYYY-MM-DD"
            m = self._year_re.search(ts_str)
            if m:
                try:
                    year = int(m.group(1))
                    if 1800 <= year <= 2030:
                        return year
                except ValueError:
                    pass

        # Fallback: scan cleaned text (title and description)
        if text_fallback:
            matches = self._year_re.findall(text_fallback)
            for match in matches:
                try:
                    year = int(match)
                    if 1800 <= year <= 2030:
                        return year
                except ValueError:
                    continue

        return None

    def extract_named_entities(self, text: str) -> List[str]:
        """Extracts candidate proper nouns, capitalized phrases, and key lexical entities."""
        if not text:
            return []

        # Find capitalized words or multi-word sequences (e.g., "Eiffel Tower", "Notre-Dame de Paris")
        entity_re = re.compile(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b")
        candidates = entity_re.findall(text)

        # Stopwords to discard
        stopwords = {
            "The", "This", "That", "There", "Here", "File", "Image", "A", "An",
            "In", "On", "At", "By", "For", "With", "About", "Against", "Between",
            "Category", "Photograph", "Photo", "Picture", "View", "Detail"
        }

        unique_entities = []
        seen = set()
        for cand in candidates:
            cand_clean = cand.strip()
            if len(cand_clean) > 2 and cand_clean not in stopwords and cand_clean.lower() not in seen:
                seen.add(cand_clean.lower())
                unique_entities.append(cand_clean)

        return unique_entities

    def parse(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        timestamp: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedMetadata:
        """Full preprocessing pipeline converting raw Commons signals into structured ParsedMetadata."""
        meta = metadata or {}
        clean_title = self.clean_title(title or meta.get("title", ""))
        clean_desc = self.clean_text(description or meta.get("description", ""))

        # Combine text for fallback signals
        combined_text = f"{clean_title}. {clean_desc}".strip()

        # Capture year extraction
        raw_ts = timestamp or meta.get("timestamp") or meta.get("DateTimeOriginal")
        year = self.extract_year(raw_ts, text_fallback=combined_text)

        # Named entities
        entities = self.extract_named_entities(combined_text)

        # GeoLocation holder (can be enriched by reverse geocoder)
        geo = None
        if latitude is not None and longitude is not None:
            geo = GeoLocation(latitude=float(latitude), longitude=float(longitude))

        return ParsedMetadata(
            title_raw=str(title or ""),
            title_clean=clean_title,
            description_raw=str(description or ""),
            description_clean=clean_desc,
            capture_year=year,
            timestamp_raw=str(raw_ts or "") if raw_ts else None,
            location=geo,
            extracted_entities=entities,
            metadata=meta,
        )
