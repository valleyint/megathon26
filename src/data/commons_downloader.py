"""Downloader for authentic Wikimedia Commons images and metadata via official API."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import urllib.request
import urllib.parse
from PIL import Image

from src.common.types import ImageQuery
from src.data.category_vocab import CategoryVocabulary
from src.data.commons_dataset import CommonsDataset


DEFAULT_CATEGORIES = [
    "Eiffel Tower",
    "Gothic architecture",
    "Taj Mahal",
    "Colosseum",
    "Steam locomotives",
    "Airbus A380",
    "Felis catus",
    "Canis lupus",
    "Paintings by Vincent van Gogh",
    "Bridges in London",
]


def is_topical_category(title: str) -> bool:
    """Filter out administrative, license, maintenance, and tracking categories."""
    clean = title.replace("Category:", "").strip().lower()
    excluded_keywords = [
        "cc-", "cc by", "gfdl", "public domain", "pd-", "self-published",
        "uploaded", "media from", "media needing", "files with", "files by",
        "images by", "created with", "taken with", "supported by",
        "featured pictures", "quality images", "valued images",
        "svg", "jpeg", "png", "tif", "audio", "video", "webm",
        "wikiproject", "wikipedia", "user:", "items with", "pages with"
    ]
    for kw in excluded_keywords:
        if kw in clean:
            return False
    return True


def fetch_commons_category_files(
    category_name: str,
    limit: int = 6,
    thumb_width: int = 400,
    user_agent: str = "MegathonX-Bot/1.0 (https://github.com/valleyint/megathon26; team23@megathon.local)",
) -> List[Dict[str, Any]]:
    """Query Wikimedia Commons API for files in a category with thumbnail URLs and tags."""
    cat_title = category_name if category_name.startswith("Category:") else f"Category:{category_name}"
    
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": cat_title,
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo|categories",
        "iiprop": "url|timestamp|extmetadata",
        "iiurlwidth": str(thumb_width),
        "clshow": "!hidden",
        "cllimit": "30",
        "format": "json",
    }
    
    url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error querying Wikimedia Commons API for '{category_name}': {e}")
        return []

    pages = data.get("query", {}).get("pages", {})
    files = []

    for page_id, page_info in pages.items():
        imageinfo = page_info.get("imageinfo", [{}])[0]
        thumb_url = imageinfo.get("thumburl") or imageinfo.get("url")
        if not thumb_url:
            continue

        # Extract only topical, non-administrative categories
        raw_cats = [
            c["title"] for c in page_info.get("categories", [])
            if "title" in c and is_topical_category(c["title"])
        ]
        if cat_title not in raw_cats and is_topical_category(cat_title):
            raw_cats.append(cat_title)

        # Extract timestamp and geolocation if present
        extmeta = imageinfo.get("extmetadata", {})
        dt = extmeta.get("DateTimeOriginal", {}).get("value") or imageinfo.get("timestamp")
        lat = extmeta.get("GPSLatitude", {}).get("value")
        lon = extmeta.get("GPSLongitude", {}).get("value")

        try:
            lat_f = float(lat) if lat else None
            lon_f = float(lon) if lon else None
        except ValueError:
            lat_f, lon_f = None, None

        files.append({
            "title": page_info.get("title", ""),
            "thumb_url": thumb_url,
            "timestamp": dt,
            "latitude": lat_f,
            "longitude": lon_f,
            "categories": raw_cats,
            "description": extmeta.get("ImageDescription", {}).get("value", ""),
        })

    return files


def download_real_commons_dataset(
    output_dir: str | Path,
    categories: List[str] = DEFAULT_CATEGORIES,
    images_per_category: int = 5,
    thumb_width: int = 400,
) -> Tuple[CommonsDataset, CategoryVocabulary]:
    """Downloads real Wikimedia Commons images and builds dataset and vocabulary."""
    out_p = Path(output_dir)
    img_dir = out_p / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    vocab = CategoryVocabulary()
    queries: List[ImageQuery] = []
    headers = {"User-Agent": "MegathonX-Bot/1.0 (https://github.com/valleyint/megathon26)"}

    print(f"Starting download of real Wikimedia Commons images across {len(categories)} categories...")

    for cat in categories:
        print(f"Fetching category: {cat}...")
        files = fetch_commons_category_files(
            category_name=cat,
            limit=images_per_category,
            thumb_width=thumb_width,
        )

        for idx, item in enumerate(files):
            # Clean safe filename
            safe_name = "".join(c for c in item["title"] if c.isalnum() or c in ("-", "_", ".")).strip()
            if not safe_name.lower().endswith((".jpg", ".jpeg", ".png")):
                safe_name = f"{safe_name}.jpg"
            img_path = img_dir / f"{idx:02d}_{safe_name}"

            # Download thumbnail
            try:
                req = urllib.request.Request(item["thumb_url"], headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(img_path, "wb") as f:
                        f.write(resp.read())

                # Validate image can be opened
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as e:
                print(f"Failed to download image {item['title']}: {e}")
                if img_path.exists():
                    img_path.unlink()
                continue

            # Register categories in vocabulary
            assigned_cats = []
            for c in item["categories"]:
                if is_topical_category(c):
                    vocab.add_category(c)
                    assigned_cats.append(vocab.normalize_name(c))

            queries.append(
                ImageQuery(
                    query_id=f"commons_{len(queries):04d}",
                    image_path=str(img_path),
                    title=item["title"],
                    description=item.get("description"),
                    timestamp=item.get("timestamp"),
                    latitude=item.get("latitude"),
                    longitude=item.get("longitude"),
                    ground_truth_categories=assigned_cats,
                )
            )
            time.sleep(0.2)  # Respect Wikimedia API rate guidelines

    dataset = CommonsDataset(queries=queries)
    dataset.save(out_p / "dataset.json")
    vocab.save(out_p / "category_vocab.json")

    print(f"\nSuccessfully downloaded {len(dataset)} real Wikimedia Commons images across {len(vocab)} unique categories.")
    print(f"Dataset saved to: {out_p / 'dataset.json'}")
    print(f"Vocabulary saved to: {out_p / 'category_vocab.json'}")
    return dataset, vocab
