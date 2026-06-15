"""Configuration utilities."""

import yaml

from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)