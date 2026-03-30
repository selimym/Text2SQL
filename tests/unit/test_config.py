import pytest
import structlog

from app.core.config import AppSettings
from app.core.logging import configure_logging


def test_defaults() -> None:
    s = AppSettings()
    assert s.log_level == "INFO"
    assert s.max_result_rows == 100
    assert s.pipeline_variant == "baseline"
    assert s.max_retries == 2
    assert s.agent_max_iterations == 10
    assert s.langsmith_project == "text2sql"
    assert s.schema_similarity_threshold is None


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LANGSMITH_PROJECT", "custom-project")
    monkeypatch.setenv("SCHEMA_SIMILARITY_THRESHOLD", "0.5")
    s = AppSettings()
    assert s.log_level == "DEBUG"
    assert s.llm_provider == "openai"
    assert s.langsmith_project == "custom-project"
    assert s.schema_similarity_threshold == 0.5


def test_configure_logging_does_not_raise() -> None:
    configure_logging("INFO")
    # structlog should be configured after calling this
    logger = structlog.get_logger()
    assert logger is not None
