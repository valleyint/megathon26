"""Common data structures and schemas used across the Commons Category pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class CategoryInfo:
    """Metadata for a Wikimedia Commons category."""
    category_id: int
    name: str
    frequency: int = 0
    is_head: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    """A scored candidate category produced by a scorer or retrieval stage."""
    category_id: int
    name: str
    score: float
    source: str  # e.g., 'knn', 'zero_shot', 'event_retrieval', 'stage1_fused'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_id": self.category_id,
            "name": self.name,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class Stage1CandidatePool:
    """The unified candidate pool produced at the end of Stage 1."""
    query_id: str
    embedding: Any  # L2-normalized image embedding z_x (np.ndarray or list)
    candidates: List[ScoredCandidate]  # Combined unique top candidates
    scorer_breakdown: Dict[str, List[ScoredCandidate]] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def category_names(self) -> List[str]:
        return [c.name for c in self.candidates]

    @property
    def category_ids(self) -> List[int]:
        return [c.category_id for c in self.candidates]


@dataclass
class GeoLocation:
    """Geographic information extracted or reverse-geocoded from geotags."""
    latitude: float
    longitude: float
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # (min_lat, min_lon, max_lat, max_lon)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "country_code": self.country_code,
            "bounding_box": self.bounding_box,
        }


@dataclass
class ParsedMetadata:
    """Parsed, cleaned wikitext metadata and EXIF signals."""
    title_raw: str = ""
    title_clean: str = ""
    description_raw: str = ""
    description_clean: str = ""
    capture_year: Optional[int] = None
    timestamp_raw: Optional[str] = None
    location: Optional[GeoLocation] = None
    extracted_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title_clean": self.title_clean,
            "description_clean": self.description_clean,
            "capture_year": self.capture_year,
            "timestamp_raw": self.timestamp_raw,
            "location": self.location.to_dict() if self.location else None,
            "extracted_entities": self.extracted_entities,
            "metadata": self.metadata,
        }


@dataclass
class PruningDecision:
    """Audit log entry explaining why a candidate category was pruned or retained."""
    category_id: int
    name: str
    status: str  # 'retained', 'suppressed_ancestor', 'temporal_conflict', 'spatial_conflict'
    reason: str = ""
    score: float = 0.0
    triggering_descendant: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_id": self.category_id,
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "score": self.score,
            "triggering_descendant": self.triggering_descendant,
            "metadata": self.metadata,
        }


@dataclass
class Stage2CandidatePool:
    """The filtered, non-redundant, grounded candidate pool produced by Stage 2."""
    query_id: str
    candidates: List[ScoredCandidate]  # Filtered candidates (10-15 target)
    pruned_decisions: List[PruningDecision] = field(default_factory=list)
    event_candidates: List[ScoredCandidate] = field(default_factory=list)
    parsed_metadata: Optional[ParsedMetadata] = None
    latency_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def category_names(self) -> List[str]:
        return [c.name for c in self.candidates]

    @property
    def category_ids(self) -> List[int]:
        return [c.category_id for c in self.candidates]

    @property
    def retained_count(self) -> int:
        return len(self.candidates)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_decisions)


@dataclass
class ImageQuery:
    """Input query container for inference."""
    query_id: str
    image_path: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    timestamp: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ground_truth_categories: List[str] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Metrics container for benchmark evaluation."""
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    mAP: float = 0.0
    mean_latency_ms: float = 0.0
    num_queries: int = 0
