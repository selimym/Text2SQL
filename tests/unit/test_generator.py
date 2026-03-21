from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.llm.generator import SQLGenerator


def test_generator_extracts_sql() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="SELECT COUNT(*) FROM singer")
    gen = SQLGenerator(llm=mock_llm)
    result = gen.generate("How many singers?")
    assert result == "SELECT COUNT(*) FROM singer"


def test_generator_strips_markdown_fences() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="```sql\nSELECT 1\n```")
    gen = SQLGenerator(llm=mock_llm)
    result = gen.generate("test")
    assert result == "SELECT 1"


def test_generator_strips_plain_fences() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="```\nSELECT 1\n```")
    gen = SQLGenerator(llm=mock_llm)
    result = gen.generate("test")
    assert result == "SELECT 1"
