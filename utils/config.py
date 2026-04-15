from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
