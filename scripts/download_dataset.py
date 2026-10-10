"""Download or generate dataset for Wikimedia Commons category suggestion."""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.commons_dataset import CommonsDataset



def main():
    parser = argparse.ArgumentParser(description="Download or generate dataset for Commons Category Suggestion")
    parser.add_argument("--output-dir", type=str, default="data/commons_split", help="Target output directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset for offline development")
    parser.add_argument("--num-categories", type=int, default=30, help="Number of categories for synthetic data")
    parser.add_argument("--num-images", type=int, default=60, help="Number of images for synthetic data")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face dataset identifier (optional)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic or args.hf_dataset is None:
        print(f"Generating synthetic dataset at {out_dir}...")
        dataset, vocab = CommonsDataset.generate_synthetic_data(
            output_dir=out_dir,
            num_categories=args.num_categories,
            num_images=args.num_images,
        )
        print(f"Successfully generated {len(dataset)} items across {len(vocab)} categories.")
    else:
        print(f"Downloading dataset from Hugging Face: {args.hf_dataset}...")
        try:
            from datasets import load_dataset
            ds = load_dataset(args.hf_dataset, split="train")
            print(f"Loaded {len(ds)} items from Hugging Face.")
        except Exception as e:
            print(f"Failed to load dataset: {e}")


if __name__ == "__main__":
    main()
