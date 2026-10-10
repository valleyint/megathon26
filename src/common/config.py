"""Configuration loader for the pipeline."""

from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None



def load_config(config_path: str | Path = "configs/stage1_default.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if yaml is None:
        raise ImportError(
            "PyYAML is required to parse YAML config files. "
            "Please install it via 'pip install pyyaml' or pass JSON configuration."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
