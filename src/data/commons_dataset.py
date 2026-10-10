"""Dataset management for Wikimedia Commons images and metadata."""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import numpy as np
except ImportError:
    np = None

from src.common.types import ImageQuery
from src.data.category_vocab import CategoryVocabulary


class CommonsDataset:
    """Manages an in-memory or on-disk collection of Wikimedia Commons queries."""

    def __init__(self, queries: Optional[List[ImageQuery]] = None):
        self.queries = queries or []

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> ImageQuery:
        return self.queries[idx]

    def save(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "query_id": q.query_id,
                "image_path": q.image_path,
                "title": q.title,
                "description": q.description,
                "timestamp": q.timestamp,
                "latitude": q.latitude,
                "longitude": q.longitude,
                "ground_truth_categories": q.ground_truth_categories,
            }
            for q in self.queries
        ]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> CommonsDataset:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = [
            ImageQuery(
                query_id=item["query_id"],
                image_path=item.get("image_path"),
                title=item.get("title"),
                description=item.get("description"),
                timestamp=item.get("timestamp"),
                latitude=item.get("latitude"),
                longitude=item.get("longitude"),
                ground_truth_categories=item.get("ground_truth_categories", []),
            )
            for item in data
        ]
        return cls(queries=queries)

    @staticmethod
    def generate_synthetic_data(
        output_dir: str | Path,
        num_categories: int = 30,
        num_images: int = 60,
    ) -> Tuple[CommonsDataset, CategoryVocabulary]:
        """Generates a synthetic Commons dataset with dummy images for testing."""
        from typing import Tuple
        out_p = Path(output_dir)
        img_dir = out_p / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        vocab = CategoryVocabulary()
        category_topics = [
            "Gothic architecture", "Rayonnant Gothic", "Romanesque churches",
            "Cathedrals in France", "Eiffel Tower", "Statue of Liberty",
            "Steam locomotives", "Electric multiple units", "High-speed rail",
            "Oil paintings", "Portrait paintings", "Impressionist art",
            "Baroque sculpture", "Flora of Europe", "Rosa (plant)",
            "Quercus robur", "Canis lupus", "Felis catus",
            "Solar eclipses", "Astronomical observatories", "Lunar photography",
            "Bridges in London", "Tower Bridge", "Golden Gate Bridge",
            "Mountain ranges of the Alps", "Mont Blanc", "Matterhorn",
            "2023 in Paris", "Strikes in France in 2023", "Festivals in Germany"
        ]

        for topic in category_topics[:num_categories]:
            vocab.add_category(topic)

        all_cat_names = vocab.get_all_names()
        queries = []

        # Color palette for synthetic images
        colors = [
            (220, 50, 50), (50, 180, 50), (50, 80, 220),
            (240, 200, 30), (180, 50, 200), (50, 200, 200)
        ]

        for i in range(num_images):
            img_id = f"wiki_img_{i:04d}"
            img_path = img_dir / f"{img_id}.jpg"

            # Create a simple colored test image
            color = colors[i % len(colors)]
            img = Image.new("RGB", (224, 224), color=color)
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 20, 204, 204], outline=(255, 255, 255), width=3)
            img.save(img_path, "JPEG")

            # Assign 1 to 3 categories
            primary_cat_idx = i % len(all_cat_names)
            assigned_cats = [all_cat_names[primary_cat_idx]]
            if (i + 1) % 2 == 0 and len(all_cat_names) > 1:
                assigned_cats.append(all_cat_names[(primary_cat_idx + 1) % len(all_cat_names)])

            queries.append(
                ImageQuery(
                    query_id=img_id,
                    image_path=str(img_path),
                    title=f"Commons File {img_id}",
                    description=f"Synthetic test image depicting {assigned_cats[0]}",
                    timestamp="2023-05-15 14:30:00",
                    latitude=48.8584,
                    longitude=2.2945,
                    ground_truth_categories=assigned_cats,
                )
            )

        dataset = CommonsDataset(queries=queries)
        dataset.save(out_p / "dataset.json")
        vocab.save(out_p / "category_vocab.json")

        return dataset, vocab
