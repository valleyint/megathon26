"""CLI script to build and serialize the FAISS Historical Image Index."""

import argparse
from pathlib import Path
from src.common.config import load_config
from src.data.category_vocab import CategoryVocabulary
from src.data.commons_dataset import CommonsDataset
from src.encoders.siglip_encoder import SigLIPVisionEncoder
from src.index_builder.build_faiss import build_faiss_index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index for Historical Images")
    parser.add_argument("--config", type=str, default="configs/stage1_default.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="data/commons_split/dataset.json", help="Path to dataset JSON")
    parser.add_argument("--vocab", type=str, default=None, help="Path to Category Vocabulary JSON")
    parser.add_argument("--mock", action="store_true", help="Use mock encoder for fast verification")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab_path = args.vocab or cfg["paths"]["vocab_path"]
    out_index_path = cfg["paths"]["faiss_index_path"]
    out_labels_path = cfg["paths"]["faiss_labels_path"]

    dataset = CommonsDataset.load(args.dataset)
    vocab = CategoryVocabulary.load(vocab_path)
    print(f"Loaded {len(dataset)} images and {len(vocab)} categories.")

    vision_encoder = SigLIPVisionEncoder(
        model_name=cfg["encoder"]["model_name"],
        torch_dtype=cfg["encoder"]["torch_dtype"],
        mock_dim=768 if args.mock else None,
    )

    build_faiss_index(
        dataset=dataset,
        vocab=vocab,
        vision_encoder=vision_encoder,
        output_index_path=out_index_path,
        output_labels_path=out_labels_path,
        temperature=cfg["scorers"]["knn"]["temperature"],
    )


if __name__ == "__main__":
    main()
