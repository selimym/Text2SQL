from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from app.llm.generator import SQLGenerator


def test_generator_extracts_sql() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="SELECT COUNT(*) FROM singer")
    gen = SQLGenerator(llm=mock_llm)
    sql, usage = gen.generate("How many singers?")
    assert sql == "SELECT COUNT(*) FROM singer"
    assert isinstance(usage, dict)


def test_generator_strips_markdown_fences() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="```sql\nSELECT 1\n```")
    gen = SQLGenerator(llm=mock_llm)
    sql, usage = gen.generate("test")
    assert sql == "SELECT 1"


def test_generator_strips_plain_fences() -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="```\nSELECT 1\n```")
    gen = SQLGenerator(llm=mock_llm)
    sql, usage = gen.generate("test")
    assert sql == "SELECT 1"


async def test_agenerate_extracts_sql() -> None:
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="SELECT COUNT(*) FROM singer"))
    gen = SQLGenerator(llm=mock_llm)
    sql, usage = await gen.agenerate("How many singers?")
    assert sql == "SELECT COUNT(*) FROM singer"
    assert isinstance(usage, dict)


async def test_agenerate_strips_markdown_fences() -> None:
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="```sql\nSELECT 1\n```"))
    gen = SQLGenerator(llm=mock_llm)
    sql, usage = await gen.agenerate("test")
    assert sql == "SELECT 1"
