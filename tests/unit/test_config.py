import pytest

from app.core.config import AppSettings


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
