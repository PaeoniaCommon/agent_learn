"""Shared settings building blocks: YAML loading with ${ENV_VAR} expansion, LLM and date settings."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class LLMSettings(BaseModel):
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0
    timeout_s: float = 60
    max_retries: int = 2
    structured_output_method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema"


class DateSettings(BaseModel):
    timezone: str = "UTC"
    day_first: bool = True
    week_start: Literal["monday", "sunday"] = "monday"
    fiscal_year_start_month: int = Field(1, ge=1, le=12)
    default_date_format: str = "%Y-%m-%d"


def merge_llm_settings(base: LLMSettings, override: LLMSettings | None) -> LLMSettings:
    """Fields set on `override` win; unset ones inherit from `base`."""
    if override is None:
        return base
    merged = base.model_dump()
    merged.update(override.model_dump(exclude_unset=True))
    return LLMSettings(**merged)


def expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default}; empty results become None."""
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        out = _ENV_RE.sub(sub, value)
        return out if out != "" else None
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_yaml(source: str | Path | dict | None) -> dict:
    """Read a YAML file (or take a dict) and expand environment variables."""
    if source is None:
        return {}
    if isinstance(source, dict):
        return expand_env(source)
    with open(source, encoding="utf-8") as fh:
        return expand_env(yaml.safe_load(fh) or {})
