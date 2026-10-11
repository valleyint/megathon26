"""Interactive Demo CLI for Stage 2 Metadata, Event Discovery, and DAG Pruning."""

import sys
import argparse
import json
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.types import Stage1CandidatePool, ScoredCandidate, ImageQuery
from src.data.category_vocab import CategoryVocabulary
from src.stage2.graph.category_dag import CategoryDAG
from src.stage2.pipeline import Stage2Pipeline


def build_sample_environment():
    """Builds an authentic sample category vocabulary and Commons DAG."""
    vocab = CategoryVocabulary()
    # Structural hierarchy
    cat_arch = vocab.add_category("Architecture")
    cat_gothic = vocab.add_category("Gothic architecture")
    cat_rayonnant = vocab.add_category("Rayonnant Gothic architecture")
    cat_cathedral = vocab.add_category("Cathedrals in France")
    cat_notredame = vocab.add_category("Notre-Dame de Paris")

    cat_tower = vocab.add_category("Towers in France")
    cat_eiffel = vocab.add_category("Eiffel Tower")
    cat_eiffel_night = vocab.add_category("Eiffel Tower at night")

    # Spatial / temporal distractors
    cat_london = vocab.add_category("Bridges in London")
    cat_colosseum = vocab.add_category("Colosseum in Rome")
    cat_old_protest = vocab.add_category("Demonstrations in Paris in 2015")

    # Target event categories
    cat_event_2023 = vocab.add_category("Demonstrations and protests in Paris in 2023")
    cat_olympics = vocab.add_category("2024 Summer Olympics in Paris")

    dag = CategoryDAG(vocab=vocab)
    for info in vocab.id_to_cat.values():
        dag.add_category(info.category_id, info.name)

    # Hierarchical links
    dag.add_edge(cat_arch, cat_gothic)
    dag.add_edge(cat_gothic, cat_rayonnant)
    dag.add_edge(cat_gothic, cat_cathedral)
    dag.add_edge(cat_cathedral, cat_notredame)

    dag.add_edge(cat_tower, cat_eiffel)
    dag.add_edge(cat_eiffel, cat_eiffel_night)

    return vocab, dag


def run_demo():
    parser = argparse.ArgumentParser(description="Demonstrate Stage 2 Filtering and DAG Pruning")
    parser.add_argument("--year", type=int, default=2023, help="EXIF capture year")
    parser.add_argument("--lat", type=float, default=48.8584, help="GPS latitude (default: Paris)")
    parser.add_argument("--lon", type=float, default=2.2945, help="GPS longitude (default: Paris)")
    parser.add_argument("--title", type=str, default="File:Night_view_of_Eiffel_Tower_during_protests_2023.jpg")
    parser.add_argument("--desc", type=str, default="{{Information|description=Protests and demonstrations in Paris at night}}")
    args = parser.parse_args()

    vocab, dag = build_sample_environment()
    pipeline = Stage2Pipeline(vocab=vocab, dag=dag)

    # Simulate Stage 1 pool with visual candidates (including redundant ancestors & false positives)
    s1_candidates = [
        ScoredCandidate(category_id=vocab.get_id("Category:Eiffel Tower at night"), name="Category:Eiffel Tower at night", score=0.91, source="knn"),
        ScoredCandidate(category_id=vocab.get_id("Category:Eiffel Tower"), name="Category:Eiffel Tower", score=0.88, source="prototype"),
        ScoredCandidate(category_id=vocab.get_id("Category:Towers in France"), name="Category:Towers in France", score=0.80, source="knn"),
        ScoredCandidate(category_id=vocab.get_id("Category:Bridges in London"), name="Category:Bridges in London", score=0.68, source="knn"),
        ScoredCandidate(category_id=vocab.get_id("Category:Demonstrations in Paris in 2015"), name="Category:Demonstrations in Paris in 2015", score=0.65, source="prototype"),
    ]

    stage1_pool = Stage1CandidatePool(
        query_id="demo_query_001",
        embedding=None,
        candidates=s1_candidates,
        latency_ms={"vision_encoding": 19.2, "knn_scorer": 4.1, "prototype_scorer": 1.8, "aggregation": 1.5},
    )

    query = ImageQuery(
        query_id="demo_query_001",
        title=args.title,
        description=args.desc,
        timestamp=f"{args.year}:05:14 21:30:00",
        latitude=args.lat,
        longitude=args.lon,
    )

    print("=" * 80)
    print("STAGE 2: METADATA, EVENT DISCOVERY & DAG PRUNING DEMO")
    print("=" * 80)
    print(f"Query Title:       {query.title}")
    print(f"Query Description: {query.description}")
    print(f"EXIF Timestamp:    {query.timestamp}")
    print(f"GPS Geotags:       Lat={query.latitude}, Lon={query.longitude}")
    print("-" * 80)

    print("\n[Stage 1 Input Candidate Pool]")
    for idx, c in enumerate(stage1_pool.candidates, 1):
        print(f"  {idx}. [{c.score:.4f}] {c.name} ({c.source})")

    # Execute Stage 2
    stage2_pool = pipeline.process(stage1_pool, query)

    meta = stage2_pool.parsed_metadata
    print("\n[Signal Extraction & Reverse Geocoding]")
    print(f"  • Clean Title:   {meta.title_clean}")
    print(f"  • Clean Desc:    {meta.description_clean}")
    print(f"  • Capture Year:  {meta.capture_year}")
    if meta.location:
        print(f"  • Resolved Geo:  {meta.location.city}, {meta.location.state}, {meta.location.country} ({meta.location.country_code})")

    print("\n[Proactive Event Discovery]")
    if stage2_pool.event_candidates:
        for idx, ev in enumerate(stage2_pool.event_candidates, 1):
            print(f"  {idx}. [{ev.score:.4f}] {ev.name} ({ev.metadata.get('match_type')})")
    else:
        print("  None discovered.")

    print("\n[Pruning Audit Trail]")
    for dec in stage2_pool.pruned_decisions:
        if dec.status != "retained":
            print(f"  • [{dec.status.upper()}] {dec.name}")
            print(f"    Reason: {dec.reason}")

    print("\n[Final Stage 2 Retained Candidates (Most-Specific & Grounded)]")
    for idx, c in enumerate(stage2_pool.candidates, 1):
        print(f"  {idx}. [{c.score:.4f}] {c.name} ({c.source})")

    print("\n[Sub-Stage Latency Profile]")
    total_latency = 0.0
    for step, ms in stage2_pool.latency_ms.items():
        total_latency += ms
        print(f"  • {step:<32}: {ms:>6.2f} ms")
    print(f"  • {'TOTAL PIPELINE LATENCY':<32}: {total_latency:>6.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
