"""RPA agent settings (YAML with ${ENV_VAR} expansion via agent_core)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from agent_core.config import DateSettings, LLMSettings, load_yaml, merge_llm_settings


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
        return merge_llm_settings(self.llm, self.learning_llm)


def load_config(source: str | Path | dict | Settings | None = None) -> Settings:
    """Load settings from a YAML path, a dict, an existing Settings, or defaults."""
    if isinstance(source, Settings):
        return source
    return Settings(**load_yaml(source))
