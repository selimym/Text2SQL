from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.models import QueryRequest
from app.db.executor import ExecutionResult, SQLExecutor
from app.db.validator import SQLValidator, ValidationResult
from app.pipeline.agent_tools import ToolContext
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


def make_agent_result(sql: str, tool_calls: int = 0) -> dict[str, Any]:
    """Create a mock agent result with the given SQL and tool call count."""
    messages: list[Any] = [HumanMessage(content="question")]
    for i in range(tool_calls):
        messages.append(ToolMessage(content="schema data", tool_call_id=f"call_{i}"))
    messages.append(AIMessage(content=sql))
    return {"messages": messages}


def make_tool_context(
    validation_valid: bool = True,
    exec_success: bool = True,
) -> ToolContext:
    mock_validator: MagicMock = MagicMock(spec=SQLValidator)
    mock_executor: MagicMock = MagicMock(spec=SQLExecutor)
    mock_validator.validate.return_value = ValidationResult(
        valid=validation_valid,
        error=None if validation_valid else "syntax error",
    )
    mock_executor.execute.return_value = ExecutionResult(
        success=exec_success,
        rows=[["1"]] if exec_success else [],
        row_count=1 if exec_success else 0,
    )
    return ToolContext(
        schema_retriever=MagicMock(spec=SchemaRetriever),
        example_retriever=MagicMock(spec=ExampleRetriever),
        validator=mock_validator,
        executor=mock_executor,
        spider_data_dir="spider_data",
    )


@patch("app.pipeline.agent_pipeline.create_react_agent")
def test_happy_path(mock_cra: MagicMock) -> None:
    from app.pipeline.agent_pipeline import AgentPipeline

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = make_agent_result("SELECT COUNT(*) FROM singers")
    mock_cra.return_value = mock_agent

    ctx = make_tool_context()
    llm = MagicMock()
    pipeline = AgentPipeline(tool_context=ctx, llm=llm, max_iterations=10)

    request = QueryRequest(question="How many singers?", db_id="concert_singer")
    response = pipeline.run(request)

    assert response.generated_sql == "SELECT COUNT(*) FROM singers"
    assert response.retry_count == 0


@patch("app.pipeline.agent_pipeline.create_react_agent")
def test_retry_count_equals_tool_messages(mock_cra: MagicMock) -> None:
    from app.pipeline.agent_pipeline import AgentPipeline

    sql = "SELECT COUNT(*) FROM singers"
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = make_agent_result(sql, tool_calls=3)
    mock_cra.return_value = mock_agent

    ctx = make_tool_context()
    llm = MagicMock()
    pipeline = AgentPipeline(tool_context=ctx, llm=llm, max_iterations=10)

    request = QueryRequest(question="How many singers?", db_id="concert_singer")
    response = pipeline.run(request)

    assert response.retry_count == 3
    assert response.generated_sql == sql


@patch("app.pipeline.agent_pipeline.create_react_agent")
def test_no_tool_calls(mock_cra: MagicMock) -> None:
    from app.pipeline.agent_pipeline import AgentPipeline

    sql = "SELECT name FROM artists"
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = make_agent_result(sql, tool_calls=0)
    mock_cra.return_value = mock_agent

    ctx = make_tool_context()
    llm = MagicMock()
    pipeline = AgentPipeline(tool_context=ctx, llm=llm, max_iterations=10)

    request = QueryRequest(question="List artists?", db_id="concert_singer")
    response = pipeline.run(request)

    assert response.generated_sql == sql
    assert response.retry_count == 0
    assert response.answer != ""


@patch("app.pipeline.agent_pipeline.create_react_agent")
def test_strips_markdown_fences(mock_cra: MagicMock) -> None:
    from app.pipeline.agent_pipeline import AgentPipeline

    raw_with_fence = "```sql\nSELECT COUNT(*) FROM singers\n```"
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = make_agent_result(raw_with_fence)
    mock_cra.return_value = mock_agent

    ctx = make_tool_context()
    llm = MagicMock()
    pipeline = AgentPipeline(tool_context=ctx, llm=llm, max_iterations=10)

    request = QueryRequest(question="How many singers?", db_id="concert_singer")
    response = pipeline.run(request)

    assert response.generated_sql == "SELECT COUNT(*) FROM singers"


@patch("app.pipeline.agent_pipeline.create_react_agent")
def test_validation_failure_sets_flag(mock_cra: MagicMock) -> None:
    from app.pipeline.agent_pipeline import AgentPipeline

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = make_agent_result("INVALID SQL")
    mock_cra.return_value = mock_agent

    ctx = make_tool_context(validation_valid=False)
    llm = MagicMock()
    pipeline = AgentPipeline(tool_context=ctx, llm=llm, max_iterations=10)

    request = QueryRequest(question="Bad query", db_id="concert_singer")
    response = pipeline.run(request)

    assert "validation_failed" in response.flags
    assert response.generated_sql == "INVALID SQL"


@patch("app.pipeline.agent_pipeline.create_react_agent")
def test_build_pipeline_agent_returns_agent_pipeline(mock_cra: MagicMock) -> None:
    from unittest.mock import MagicMock

    from app.pipeline.agent_pipeline import AgentPipeline
    from app.pipeline.factory import build_pipeline

    mock_cra.return_value = MagicMock()

    schema_retriever = MagicMock(spec=SchemaRetriever)
    example_retriever = MagicMock(spec=ExampleRetriever)
    assembler = MagicMock()
    generator = MagicMock()
    generator.llm = MagicMock()
    validator = MagicMock(spec=SQLValidator)
    executor = MagicMock(spec=SQLExecutor)

    pipeline = build_pipeline(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
        variant="agent",
    )

    assert isinstance(pipeline, AgentPipeline)
