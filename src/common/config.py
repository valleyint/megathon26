"""Configuration loader for the pipeline."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path = "configs/stage1_default.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
