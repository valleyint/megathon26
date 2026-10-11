"""Stage 2 End-to-End Orchestrator Pipeline: Metadata Preprocessing, Event Discovery, and DAG Pruning."""

from __future__ import annotations
from typing import Dict, Any, Optional, Union, List

from src.common.types import (
    Stage1CandidatePool,
    Stage2CandidatePool,
    ImageQuery,
    ScoredCandidate,
    ParsedMetadata,
    PruningDecision,
)
from src.common.timer import PipelineTimer
from src.data.category_vocab import CategoryVocabulary
from src.stage2.metadata.cleaner import WikitextMetadataCleaner
from src.stage2.metadata.geocoder import OfflineReverseGeocoder
from src.stage2.events.hypothesis_generator import EventHypothesisGenerator
from src.stage2.events.event_retriever import EventIndexRetriever
from src.stage2.filters.temporal_filter import TemporalConflictFilter
from src.stage2.filters.spatial_filter import SpatialConflictFilter
from src.stage2.graph.category_dag import CategoryDAG
from src.stage2.graph.ancestor_suppressor import AncestorSuppressor


class Stage2Pipeline:
    """Orchestrates Stage 2 metadata parsing, proactive event discovery, conflict pruning, and DAG restructuring."""

    def __init__(
        self,
        vocab: CategoryVocabulary,
        dag: Optional[CategoryDAG] = None,
        cleaner: Optional[WikitextMetadataCleaner] = None,
        geocoder: Optional[OfflineReverseGeocoder] = None,
        event_generator: Optional[EventHypothesisGenerator] = None,
        event_retriever: Optional[EventIndexRetriever] = None,
        temporal_filter: Optional[TemporalConflictFilter] = None,
        spatial_filter: Optional[SpatialConflictFilter] = None,
        ancestor_suppressor: Optional[AncestorSuppressor] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.vocab = vocab
        self.config = config or {}

        # Component initialization with sensible defaults
        self.cleaner = cleaner or WikitextMetadataCleaner()
        self.geocoder = geocoder or OfflineReverseGeocoder()
        self.dag = dag or CategoryDAG(vocab=vocab)

        self.event_generator = event_generator or EventHypothesisGenerator()
        self.event_retriever = event_retriever or EventIndexRetriever(vocab=self.vocab)

        year_tol = self.config.get("filters", {}).get("temporal", {}).get("year_tolerance", 1)
        self.temporal_filter = temporal_filter or TemporalConflictFilter(year_tolerance=year_tol)

        strict_spatial = self.config.get("filters", {}).get("spatial", {}).get("strict_country_check", True)
        self.spatial_filter = spatial_filter or SpatialConflictFilter(
            geocoder=self.geocoder, strict_country_check=strict_spatial
        )

        child_thresh = self.config.get("dag_pruning", {}).get("child_confidence_threshold", 0.35)
        min_retained = self.config.get("stage2", {}).get("min_pool_size", 5)
        self.ancestor_suppressor = ancestor_suppressor or AncestorSuppressor(
            dag=self.dag,
            child_confidence_threshold=child_thresh,
            min_retained_candidates=min_retained,
        )

        self.target_pool_size = self.config.get("stage2", {}).get("target_pool_size", 15)

    def process(
        self,
        stage1_pool: Stage1CandidatePool,
        query: Optional[Union[ImageQuery, Dict[str, Any]]] = None,
    ) -> Stage2CandidatePool:
        """Processes a Stage 1 candidate pool through the complete Stage 2 pipeline."""
        timer = PipelineTimer()
        all_pruned_decisions: List[PruningDecision] = []

        # 1. Metadata Preprocessing & Geocoding
        with timer.measure("metadata_preprocessing"):
            q_title = getattr(query, "title", None) if query else None
            q_desc = getattr(query, "description", None) if query else None
            q_time = getattr(query, "timestamp", None) if query else None
            q_lat = getattr(query, "latitude", None) if query else None
            q_lon = getattr(query, "longitude", None) if query else None

            if isinstance(query, dict):
                q_title = query.get("title")
                q_desc = query.get("description")
                q_time = query.get("timestamp")
                q_lat = query.get("latitude")
                q_lon = query.get("longitude")

            parsed_meta = self.cleaner.parse(
                title=q_title,
                description=q_desc,
                timestamp=q_time,
                latitude=q_lat,
                longitude=q_lon,
            )

            # Reverse geocode if coordinates present
            if parsed_meta.location and parsed_meta.location.latitude is not None:
                enriched_geo = self.geocoder.reverse_geocode(
                    parsed_meta.location.latitude,
                    parsed_meta.location.longitude,
                )
                if enriched_geo:
                    parsed_meta.location = enriched_geo

        # 2. Proactive Event Discovery & Hypothesis Snapping
        event_candidates: List[ScoredCandidate] = []
        with timer.measure("event_discovery"):
            hypotheses = self.event_generator.generate_hypotheses(
                visual_anchors=stage1_pool.candidates,
                parsed_metadata=parsed_meta,
                max_hypotheses=20,
            )
            event_candidates = self.event_retriever.retrieve_and_snap(
                hypotheses=hypotheses,
                top_k=20,
            )

        # 3. Candidate Pool Ingestion (C_pool = C_visual ∪ C_event)
        with timer.measure("candidate_merging"):
            candidate_dict: Dict[int, ScoredCandidate] = {}

            # Add visual candidates from Stage 1
            for cand in stage1_pool.candidates:
                candidate_dict[cand.category_id] = cand

            # Merge event candidates
            for cand in event_candidates:
                if cand.category_id not in candidate_dict:
                    candidate_dict[cand.category_id] = cand
                else:
                    # Boost score if candidate appears in both visual and event retrieval
                    existing = candidate_dict[cand.category_id]
                    boosted_score = max(existing.score, cand.score) + 0.1 * min(existing.score, cand.score)
                    candidate_dict[cand.category_id] = ScoredCandidate(
                        category_id=cand.category_id,
                        name=cand.name,
                        score=boosted_score,
                        source="visual_and_event",
                        metadata={**existing.metadata, **cand.metadata},
                    )

            current_candidates = list(candidate_dict.values())

        # 4. Temporal Conflict Filtering
        with timer.measure("temporal_filtering"):
            current_candidates, temporal_decisions = self.temporal_filter.filter(
                candidates=current_candidates,
                parsed_metadata=parsed_meta,
            )
            all_pruned_decisions.extend(temporal_decisions)

        # 5. Spatial Conflict Filtering
        with timer.measure("spatial_filtering"):
            current_candidates, spatial_decisions = self.spatial_filter.filter(
                candidates=current_candidates,
                parsed_metadata=parsed_meta,
            )
            all_pruned_decisions.extend(spatial_decisions)

        # 6. Category DAG Ancestor Suppression ("Most-Specific" Rule)
        with timer.measure("dag_ancestor_pruning"):
            current_candidates, dag_decisions = self.ancestor_suppressor.suppress(
                candidates=current_candidates,
            )
            all_pruned_decisions.extend(dag_decisions)

        # 7. Final Pool Restriction (10-15 categories)
        with timer.measure("pool_sizing"):
            # Sort by score descending
            current_candidates.sort(key=lambda x: x.score, reverse=True)
            final_candidates = current_candidates[: self.target_pool_size]

        summary_latency = {
            **stage1_pool.latency_ms,
            **{f"stage2_{k}": v for k, v in timer.get_summary().items()},
        }

        return Stage2CandidatePool(
            query_id=stage1_pool.query_id,
            candidates=final_candidates,
            pruned_decisions=all_pruned_decisions,
            event_candidates=event_candidates,
            parsed_metadata=parsed_meta,
            latency_ms=summary_latency,
        )
