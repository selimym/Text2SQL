from unittest.mock import AsyncMock, MagicMock

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


async def test_agenerate_extracts_sql() -> None:
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="SELECT COUNT(*) FROM singer"))
    gen = SQLGenerator(llm=mock_llm)
    result = await gen.agenerate("How many singers?")
    assert result == "SELECT COUNT(*) FROM singer"


async def test_agenerate_strips_markdown_fences() -> None:
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="```sql\nSELECT 1\n```"))
    gen = SQLGenerator(llm=mock_llm)
    result = await gen.agenerate("test")
    assert result == "SELECT 1"
