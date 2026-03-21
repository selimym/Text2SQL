from unittest.mock import patch

import pytest

from app.core.llm import get_llm


def test_get_llm_anthropic() -> None:
    with patch("app.core.llm.ChatAnthropic") as mock_cls:
        get_llm("anthropic", "claude-3-5-sonnet-20241022")
        mock_cls.assert_called_once_with(model="claude-3-5-sonnet-20241022")


def test_get_llm_openai() -> None:
    with patch("app.core.llm.ChatOpenAI") as mock_cls:
        get_llm("openai", "gpt-4o")
        mock_cls.assert_called_once_with(model="gpt-4o")


def test_get_llm_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm("unknown", "some-model")
