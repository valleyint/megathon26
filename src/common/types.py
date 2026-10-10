"""Common data structures and schemas used across the Commons Category pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
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
    """A scored candidate category produced by a scorer."""
    category_id: int
    name: str
    score: float
    source: str  # e.g., 'knn', 'zero_shot', 'supervised'
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
    embedding: np.ndarray  # L2-normalized image embedding z_x
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
