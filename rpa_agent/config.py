"""Configuration: YAML file with ${ENV_VAR} expansion → pydantic settings."""

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


class PathSettings(BaseModel):
    knowledge_dir: str = "./knowledge"


class AgentSettings(BaseModel):
    view_shortlist_max: int = 40
    fuzzy_match_cutoff: float = 0.92
    max_param_reasks: int = 2
    max_view_reasks: int = 2
    sample_value_max_chars: int = 60
    max_columns_listed: int = 30
    max_options_listed: int = 10
    max_valid_values_in_prompt: int = 30


class LearningSettings(BaseModel):
    enabled: bool = True
    view_desc_max_chars: int = 200
    param_desc_max_chars: int = 150
    param_format_max_chars: int = 250
    max_examples: int = 3
    debounce_s: float = 2
    conflict_threshold: int = 2
    recent_evidence_lines: int = 10


class DataStoreSettings(BaseModel):
    max_entries: int = 50


class DateSettings(BaseModel):
    timezone: str = "UTC"
    day_first: bool = True
    week_start: Literal["monday", "sunday"] = "monday"
    fiscal_year_start_month: int = Field(1, ge=1, le=12)
    default_date_format: str = "%Y-%m-%d"


class RespondSettings(BaseModel):
    stream_chunk_chars: int = 16
    respond_with_llm: bool = False


class BackendSettings(BaseModel):
    module: str | None = None  # dotted module exposing RPA_VIEWS, validate_view, ...


class Settings(BaseModel):
    llm: LLMSettings = LLMSettings()
    learning_llm: LLMSettings | None = None
    paths: PathSettings = PathSettings()
    agent: AgentSettings = AgentSettings()
    learning: LearningSettings = LearningSettings()
    data_store: DataStoreSettings = DataStoreSettings()
    dates: DateSettings = DateSettings()
    respond: RespondSettings = RespondSettings()
    backend: BackendSettings = BackendSettings()

    @property
    def effective_learning_llm(self) -> LLMSettings:
        if self.learning_llm is None:
            return self.llm
        # unset fields of learning_llm inherit from llm
        merged = self.llm.model_dump()
        merged.update(self.learning_llm.model_dump(exclude_unset=True))
        return LLMSettings(**merged)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        out = _ENV_RE.sub(sub, value)
        return out if out != "" else None
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(source: str | Path | dict | Settings | None = None) -> Settings:
    """Load settings from a YAML path, a dict, an existing Settings, or defaults."""
    if isinstance(source, Settings):
        return source
    if source is None:
        return Settings()
    if isinstance(source, dict):
        raw = source
    else:
        with open(source, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return Settings(**_expand(raw))
