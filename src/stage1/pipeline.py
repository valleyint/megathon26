"""Stage 1 End-to-End Candidate Generation Pipeline."""

from __future__ import annotations
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np

from src.encoders.base import BaseVisionEncoder
from src.stage1.scorers.base import BaseScorer
from src.stage1.aggregator import Stage1Aggregator
from src.common.types import Stage1CandidatePool, ImageQuery
from src.common.timer import PipelineTimer


class Stage1Pipeline:
    """End-to-End Orchestrator for Stage 1 Multi-Scorer Retrieval."""

    def __init__(
        self,
        vision_encoder: BaseVisionEncoder,
        scorers: Dict[str, BaseScorer],
        aggregator: Optional[Stage1Aggregator] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.vision_encoder = vision_encoder
        self.scorers = scorers
        self.aggregator = aggregator or Stage1Aggregator()
        self.config = config or {}

    def predict(
        self,
        image: Union[Image.Image, str, np.ndarray],
        query_id: str = "query_0",
        top_k_per_scorer: int = 50,
    ) -> Stage1CandidatePool:
        """Run complete Stage 1 retrieval on a single image query."""
        timer = PipelineTimer()

        # Step 1: Compute normalized vision embedding z_x once
        with timer.measure("vision_encoding"):
            z_x = self.vision_encoder.encode_image(image)

        # Step 2: Query active scorers concurrently or sequentially
        scorer_outputs = {}
        for scorer_name, scorer in self.scorers.items():
            with timer.measure(f"scorer_{scorer_name}"):
                scorer_outputs[scorer_name] = scorer.score(z_x, top_k=top_k_per_scorer)

        # Step 3: Reciprocal Rank Fusion & Aggregation
        with timer.measure("aggregation"):
            candidate_pool = self.aggregator.aggregate(
                query_id=query_id,
                embedding=z_x,
                scorer_outputs=scorer_outputs,
                latency_ms=timer.get_summary(),
            )

        return candidate_pool
