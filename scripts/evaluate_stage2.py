"""Benchmark Evaluation script for Stage 2 (DAG Pruning, Consistency Filters, and Event Discovery)."""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Set

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.types import Stage1CandidatePool, ScoredCandidate, ImageQuery
from src.data.category_vocab import CategoryVocabulary
from src.stage2.graph.category_dag import CategoryDAG
from src.stage2.pipeline import Stage2Pipeline
from src.evaluation.metrics import compute_metrics


def generate_evaluation_suite(vocab: CategoryVocabulary, dag: CategoryDAG) -> List[Dict[str, Any]]:
    """Generates a rich benchmark suite of queries reflecting Wikimedia Commons edge cases."""
    queries = [
        # Query 1: Paris Gothic Cathedral (Tests ancestor suppression + spatial filter)
        {
            "query": ImageQuery(
                query_id="eval_01",
                title="File:Notre_Dame_de_Paris_West_Facade_2022.jpg",
                description="{{Information|description=Gothic facade of Notre Dame Cathedral in Paris}}",
                timestamp="2022:06:10 14:00:00",
                latitude=48.8530,
                longitude=2.3499,
                ground_truth_categories=["Category:Notre-Dame de Paris", "Category:Gothic cathedrals in France"],
            ),
            "s1_candidates": [
                ("Category:Architecture", 0.94),
                ("Category:Gothic architecture", 0.91),
                ("Category:Gothic cathedrals in France", 0.89),
                ("Category:Notre-Dame de Paris", 0.88),
                ("Category:Bridges in London", 0.70),  # Spatial error
                ("Category:Events in 2017 in Paris", 0.65),  # Temporal error
            ],
        },
        # Query 2: Eiffel Tower Event (Tests event synthesis + year matching + ancestor pruning)
        {
            "query": ImageQuery(
                query_id="eval_02",
                title="File:Protest_against_pension_reform_at_Eiffel_Tower_2023.jpg",
                description="{{Information|description=Protests in Paris near Eiffel Tower in 2023}}",
                timestamp="2023:03:15 11:30:00",
                latitude=48.8584,
                longitude=2.2945,
                ground_truth_categories=[
                    "Category:Eiffel Tower at night",
                    "Category:Demonstrations and protests in Paris in 2023",
                ],
            ),
            "s1_candidates": [
                ("Category:Towers in France", 0.92),
                ("Category:Eiffel Tower", 0.89),
                ("Category:Eiffel Tower at night", 0.87),
                ("Category:Colosseum in Rome", 0.72),  # Spatial error
                ("Category:Demonstrations in Paris in 2015", 0.68),  # Temporal error
            ],
        },
        # Query 3: London Bridge (Tests spatial consistency in UK)
        {
            "query": ImageQuery(
                query_id="eval_03",
                title="File:Tower_Bridge_crossing_River_Thames_2021.jpg",
                description="{{Information|description=Bridges in London over River Thames}}",
                timestamp="2021:09:20 16:15:00",
                latitude=51.5055,
                longitude=-0.0754,
                ground_truth_categories=["Category:Bridges in London"],
            ),
            "s1_candidates": [
                ("Category:Bridges in London", 0.93),
                ("Category:Eiffel Tower", 0.78),  # Spatial error
                ("Category:Architecture", 0.85),
                ("Category:2024 Summer Olympics in Paris", 0.60),  # Temporal + Spatial error
            ],
        },
    ]
    return queries


def compute_redundancy_rate(predictions: List[str], dag: CategoryDAG, vocab: CategoryVocabulary) -> float:
    """Calculates the proportion of predictions that are redundant broad ancestors of another predicted category."""
    if len(predictions) <= 1:
        return 0.0

    pred_ids = [vocab.get_id(name) for name in predictions if vocab.get_id(name) is not None]
    pred_set = set(pred_ids)

    redundant_count = 0
    for p_id in pred_ids:
        descendants = dag.get_descendants(p_id)
        if descendants & pred_set:
            redundant_count += 1

    return redundant_count / len(pred_ids) if pred_ids else 0.0


def run_benchmark():
    # Build vocabulary and DAG
    from scripts.demo_stage2 import build_sample_environment

    vocab, dag = build_sample_environment()
    pipeline = Stage2Pipeline(vocab=vocab, dag=dag)

    eval_suite = generate_evaluation_suite(vocab, dag)

    s1_all_preds: List[List[str]] = []
    s2_all_preds: List[List[str]] = []
    all_ground_truths: List[List[str]] = []

    s1_redundancies = []
    s2_redundancies = []
    stage2_latencies = []

    for item in eval_suite:
        q = item["query"]
        all_ground_truths.append(q.ground_truth_categories)

        # Build Stage 1 candidate pool
        s1_cands = [
            ScoredCandidate(
                category_id=vocab.get_id(name) or 0,
                name=name,
                score=score,
                source="stage1_fused",
            )
            for name, score in item["s1_candidates"]
        ]
        s1_pool = Stage1CandidatePool(
            query_id=q.query_id,
            embedding=None,
            candidates=s1_cands,
            latency_ms={"stage1_total": 24.5},
        )

        s1_preds = s1_pool.category_names
        s1_all_preds.append(s1_preds)
        s1_redundancies.append(compute_redundancy_rate(s1_preds[:5], dag, vocab))

        # Process Stage 2
        t0 = time.perf_counter()
        s2_pool = pipeline.process(s1_pool, q)
        t1 = time.perf_counter()
        stage2_latencies.append((t1 - t0) * 1000.0)

        s2_preds = s2_pool.category_names
        s2_all_preds.append(s2_preds)
        s2_redundancies.append(compute_redundancy_rate(s2_preds[:5], dag, vocab))

    # Evaluate metrics
    k_list = [1, 3, 5]
    gt_sets = [set(gt) for gt in all_ground_truths]
    s1_metrics = compute_metrics(s1_all_preds, gt_sets, k_values=k_list)
    s2_metrics = compute_metrics(s2_all_preds, gt_sets, k_values=k_list)

    s1_map = s1_metrics.mAP
    s2_map = s2_metrics.mAP

    print("=" * 80)
    print("STAGE 1 VS. STAGE 2 RETRIEVAL & PRUNING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Stage 1 (Baseline)':<20} | {'Stage 2 (Filtered & Pruned)':<25}")
    print("-" * 80)

    for k in k_list:
        print(f"P@{k:<22} | {s1_metrics.precision_at_k[k]:<20.4f} | {s2_metrics.precision_at_k[k]:<25.4f}")
        print(f"R@{k:<22} | {s1_metrics.recall_at_k[k]:<20.4f} | {s2_metrics.recall_at_k[k]:<25.4f}")
        print(f"F1@{k:<21} | {s1_metrics.f1_at_k[k]:<20.4f} | {s2_metrics.f1_at_k[k]:<25.4f}")
        print("-" * 80)

    print(f"{'mAP':<25} | {s1_map:<20.4f} | {s2_map:<25.4f}")
    avg_s1_red = sum(s1_redundancies) / len(s1_redundancies)
    avg_s2_red = sum(s2_redundancies) / len(s2_redundancies)
    print(f"{'Ancestor Redundancy Rate':<25} | {avg_s1_red * 100.0:>18.1f}% | {avg_s2_red * 100.0:>23.1f}%")

    avg_s2_lat = sum(stage2_latencies) / len(stage2_latencies)
    print(f"{'Stage 2 Processing Latency':<25} | {'N/A':<20} | {avg_s2_lat:>21.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
