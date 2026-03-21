import pytest
import structlog

from app.core.config import AppSettings
from app.core.logging import configure_logging


def test_defaults() -> None:
    s = AppSettings()
    assert s.log_level == "INFO"
    assert s.max_result_rows == 100


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    s = AppSettings()
    assert s.log_level == "DEBUG"
    assert s.llm_provider == "openai"


def test_configure_logging_does_not_raise() -> None:
    configure_logging("INFO")
    # structlog should be configured after calling this
    logger = structlog.get_logger()
    assert logger is not None
