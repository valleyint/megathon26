"""CLI script to generate and save Zero-Shot Category Prototypes."""

import argparse
from pathlib import Path
from src.common.config import load_config
from src.data.category_vocab import CategoryVocabulary
from src.encoders.siglip_encoder import SigLIPTextEncoder
from src.index_builder.build_prototypes import build_category_prototypes


def main():
    parser = argparse.ArgumentParser(description="Precompute Category Text Prototypes")
    parser.add_argument("--config", type=str, default="configs/stage1_default.yaml", help="Path to config file")
    parser.add_argument("--vocab", type=str, default=None, help="Path to Category Vocabulary JSON")
    parser.add_argument("--output", type=str, default=None, help="Output prototypes .npy file path")
    parser.add_argument("--mock", action="store_true", help="Use mock encoder for fast verification")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab_path = args.vocab or cfg["paths"]["vocab_path"]
    out_path = args.output or cfg["paths"]["prototypes_path"]

    vocab = CategoryVocabulary.load(vocab_path)
    print(f"Loaded vocabulary with {len(vocab)} categories from {vocab_path}")

    encoder = SigLIPTextEncoder(
        model_name=cfg["encoder"]["model_name"],
        torch_dtype=cfg["encoder"]["torch_dtype"],
        mock_dim=768 if args.mock else None,
    )

    build_category_prototypes(
        vocab=vocab,
        text_encoder=encoder,
        output_path=out_path,
        batch_size=cfg["encoder"].get("batch_size", 64),
    )


if __name__ == "__main__":
    main()
