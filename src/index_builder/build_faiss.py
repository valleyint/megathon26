"""Offline builder for FAISS Historical Image Index."""

from __future__ import annotations
from pathlib import Path
from typing import Union, List
import numpy as np

from src.data.commons_dataset import CommonsDataset
from src.data.category_vocab import CategoryVocabulary
from src.encoders.base import BaseVisionEncoder
from src.stage1.scorers.knn_scorer import KNNScorer


def build_faiss_index(
    dataset: CommonsDataset,
    vocab: CategoryVocabulary,
    vision_encoder: BaseVisionEncoder,
    output_index_path: Union[str, Path],
    output_labels_path: Union[str, Path],
    temperature: float = 0.07,
) -> KNNScorer:
    """Encodes all training images, associates their categories, and builds the KNN index."""
    scorer = KNNScorer(temperature=temperature, vocab=vocab)

    image_paths: List[str] = []
    image_ids: List[str] = []
    labels_per_image: List[List[int]] = []

    for item in dataset:
        if item.image_path and Path(item.image_path).exists():
            image_paths.append(item.image_path)
            image_ids.append(item.query_id)
            cat_ids = [vocab.get_id(c) for c in item.ground_truth_categories if vocab.get_id(c) is not None]
            labels_per_image.append(cat_ids)

    print(f"Encoding {len(image_paths)} images with {vision_encoder.__class__.__name__}...")
    vectors = vision_encoder.encode_batch(image_paths)

    scorer.add_historical_images(
        vectors=vectors,
        image_ids=image_ids,
        category_ids_per_image=labels_per_image,
    )

    scorer.save(output_index_path, output_labels_path)
    print(f"Saved FAISS index to {output_index_path} and labels to {output_labels_path}")
    return scorer
