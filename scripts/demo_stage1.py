"""Interactive demo CLI for Stage 1 Category Retrieval."""

import argparse
from pathlib import Path
from src.common.config import load_config
from src.data.category_vocab import CategoryVocabulary
from src.encoders.siglip_encoder import SigLIPVisionEncoder
from src.stage1.scorers.knn_scorer import KNNScorer
from src.stage1.scorers.prototype_scorer import PrototypeScorer
from src.stage1.aggregator import Stage1Aggregator
from src.stage1.pipeline import Stage1Pipeline


def main():
    parser = argparse.ArgumentParser(description="Query Stage 1 with an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input query image")
    parser.add_argument("--config", type=str, default="configs/stage1_default.yaml", help="Path to config file")
    parser.add_argument("--mock", action="store_true", help="Use mock mode")
    parser.add_argument("--top-k", type=int, default=10, help="Number of suggestions to display")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab = CategoryVocabulary.load(cfg["paths"]["vocab_path"])

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

    print(f"\nProcessing query image: {args.image}...")
    pool = pipeline.predict(image=args.image, query_id="demo_query")

    print("\n" + "=" * 65)
    print("STAGE 1 RETRIEVAL SUGGESTIONS")
    print("=" * 65)
    for rank, cand in enumerate(pool.candidates[: args.top_k], start=1):
        exemplars = cand.metadata.get("neighbor_exemplars", [])
        ex_str = f" [Exemplars: {', '.join(exemplars)}]" if exemplars else ""
        print(f" {rank:2d}. {cand.name:<40} (Score: {cand.score:.4f}){ex_str}")

    print("-" * 65)
    print("LATENCY BREAKDOWN:")
    for step, ms in pool.latency_ms.items():
        print(f"  • {step:<25}: {ms:.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
