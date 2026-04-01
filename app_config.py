import os
from typing import Any

import yaml

from config_types import AgentConfig
from model_runner import normalize_path


def load_config(path: str) -> AgentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    model_override = os.environ.get("CELESTE_MODEL_PATH")
    if model_override:
        raw["model_path"] = model_override
    cfg = AgentConfig(**raw)
    cfg.model_path = normalize_path(cfg.model_path, base_dir=os.path.dirname(path))
    return cfg


def load_raw_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(path: str, cfg: AgentConfig, *, preserve_unknown: bool = True) -> None:
    raw: dict[str, Any] = load_raw_config(path) if preserve_unknown and os.path.exists(path) else {}
    raw.update(cfg.model_dump())
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=False)
