"""Category Vocabulary management for Wikimedia Commons."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
from src.common.types import CategoryInfo


class CategoryVocabulary:
    """Manages bijective mapping between category IDs and canonical Commons category titles."""

    def __init__(self):
        self.id_to_cat: Dict[int, CategoryInfo] = {}
        self.name_to_id: Dict[str, int] = {}

    def add_category(self, name: str, category_id: Optional[int] = None, frequency: int = 0) -> int:
        clean_name = self.normalize_name(name)
        if clean_name in self.name_to_id:
            cat_id = self.name_to_id[clean_name]
            self.id_to_cat[cat_id].frequency += frequency
            return cat_id

        if category_id is None:
            category_id = len(self.id_to_cat)

        cat_info = CategoryInfo(
            category_id=category_id,
            name=clean_name,
            frequency=frequency,
        )
        self.id_to_cat[category_id] = cat_info
        self.name_to_id[clean_name] = category_id
        return category_id

    @staticmethod
    def normalize_name(name: str) -> str:
        name = name.strip()
        if not name.startswith("Category:"):
            name = f"Category:{name}"
        return name

    def get_id(self, name: str) -> Optional[int]:
        return self.name_to_id.get(self.normalize_name(name))

    def get_name(self, category_id: int) -> Optional[str]:
        info = self.id_to_cat.get(category_id)
        return info.name if info else None

    def get_info(self, category_id: int) -> Optional[CategoryInfo]:
        return self.id_to_cat.get(category_id)

    def get_all_names(self) -> List[str]:
        return [self.id_to_cat[i].name for i in range(len(self.id_to_cat))]

    def __len__(self) -> int:
        return len(self.id_to_cat)

    def save(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "categories": [
                {
                    "category_id": cat.category_id,
                    "name": cat.name,
                    "frequency": cat.frequency,
                    "is_head": cat.is_head,
                    "metadata": cat.metadata,
                }
                for cat in self.id_to_cat.values()
            ]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str | Path) -> CategoryVocabulary:
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("categories", []):
            cat_info = CategoryInfo(
                category_id=item["category_id"],
                name=item["name"],
                frequency=item.get("frequency", 0),
                is_head=item.get("is_head", False),
                metadata=item.get("metadata", {}),
            )
            vocab.id_to_cat[cat_info.category_id] = cat_info
            vocab.name_to_id[cat_info.name] = cat_info.category_id
        return vocab
