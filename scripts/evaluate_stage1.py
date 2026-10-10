"""Evaluate Stage 1 Retrieval on a validation split."""

import argparse
from pathlib import Path
from typing import List, Set
import numpy as np

from src.common.config import load_config
from src.data.category_vocab import CategoryVocabulary
from src.data.commons_dataset import CommonsDataset
from src.encoders.siglip_encoder import SigLIPVisionEncoder
from src.stage1.scorers.knn_scorer import KNNScorer
from src.stage1.scorers.prototype_scorer import PrototypeScorer
from src.stage1.aggregator import Stage1Aggregator
from src.stage1.pipeline import Stage1Pipeline
from src.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Multi-Scorer Retrieval")
    parser.add_argument("--config", type=str, default="configs/stage1_default.yaml", help="Path to config file")
    parser.add_argument("--val-dataset", type=str, default="data/commons_split/dataset.json", help="Validation dataset JSON")
    parser.add_argument("--vocab", type=str, default=None, help="Path to Category Vocabulary JSON")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for quick integration check")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum validation samples to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab_path = args.vocab or cfg["paths"]["vocab_path"]
    vocab = CategoryVocabulary.load(vocab_path)
    dataset = CommonsDataset.load(args.val_dataset)

    print(f"Loaded {len(dataset)} validation samples and {len(vocab)} categories.")

    vision_encoder = SigLIPVisionEncoder(
        model_name=cfg["encoder"]["model_name"],
        mock_dim=768 if args.mock else None,
    )

    scorers = {}
    if cfg["scorers"]["prototype"]["enabled"]:
        scorers["prototype"] = PrototypeScorer(
            vocab=vocab,
            prototypes_path=cfg["paths"]["prototypes_path"],
        )

    if cfg["scorers"]["knn"]["enabled"]:
        scorers["knn"] = KNNScorer(
            temperature=cfg["scorers"]["knn"]["temperature"],
            top_k_neighbors=cfg["scorers"]["knn"]["top_k_neighbors"],
            vocab=vocab,
            index_path=cfg["paths"]["faiss_index_path"],
            labels_path=cfg["paths"]["faiss_labels_path"],
        )

    aggregator = Stage1Aggregator(pool_size=cfg["aggregator"]["pool_size"])
    pipeline = Stage1Pipeline(vision_encoder=vision_encoder, scorers=scorers, aggregator=aggregator)

    predictions: List[List[str]] = []
    ground_truths: List[Set[str]] = []
    latencies: List[float] = []

    samples_to_eval = min(len(dataset), args.max_samples)
    print(f"Evaluating Stage 1 on {samples_to_eval} queries...")

    for i in range(samples_to_eval):
        query = dataset[i]
        if not query.image_path or not Path(query.image_path).exists():
            continue

        res = pipeline.predict(
            image=query.image_path,
            query_id=query.query_id,
            top_k_per_scorer=cfg["aggregator"]["pool_size"],
        )

        predictions.append(res.category_names)
        ground_truths.append(set(query.ground_truth_categories))
        latencies.append(res.latency_ms.get("total_ms", 0.0))

    metrics = compute_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        k_values=(1, 5, 10, 20, 50),
        latencies_ms=latencies,
    )

    print("\n" + "=" * 50)
    print("STAGE 1 RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Evaluated Queries: {metrics.num_queries}")
    print(f"Mean Latency:      {metrics.mean_latency_ms:.2f} ms")
    print(f"mAP:               {metrics.mAP:.4f}")
    print("-" * 50)
    print(f"{'k':<5} | {'Precision@k':<12} | {'Recall@k':<10} | {'F1@k':<8}")
    print("-" * 50)
    for k in (1, 5, 10, 20, 50):
        p = metrics.precision_at_k.get(k, 0.0)
        r = metrics.recall_at_k.get(k, 0.0)
        f1 = metrics.f1_at_k.get(k, 0.0)
        print(f"{k:<5} | {p:<12.4f} | {r:<10.4f} | {f1:<8.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
