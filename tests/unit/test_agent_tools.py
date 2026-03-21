import json
from unittest.mock import MagicMock

from app.db.executor import ExecutionResult, SQLExecutor
from app.db.schema_doc import ColumnInfo, SchemaDocument
from app.db.validator import SQLValidator, ValidationResult
from app.pipeline.agent_tools import ToolContext, make_tools
from app.retrieval.example_doc import ExampleDocument
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


def make_context() -> ToolContext:
    schema_retriever = MagicMock(spec=SchemaRetriever)
    example_retriever = MagicMock(spec=ExampleRetriever)
    validator = MagicMock(spec=SQLValidator)
    executor = MagicMock(spec=SQLExecutor)
    return ToolContext(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
    )


def test_get_schema_tool_returns_formatted_text() -> None:
    ctx = make_context()
    docs = [
        SchemaDocument(
            db_id="mydb",
            table_name="users",
            columns=[ColumnInfo(name="id", data_type="INTEGER")],
            foreign_keys=[],
        ),
        SchemaDocument(
            db_id="mydb",
            table_name="orders",
            columns=[ColumnInfo(name="order_id", data_type="INTEGER")],
            foreign_keys=[],
        ),
    ]
    ctx.schema_retriever.retrieve.return_value = docs  # type: ignore[attr-defined]

    tools = make_tools(ctx)
    get_schema = next(t for t in tools if t.name == "get_schema")
    result = get_schema.invoke({"question": "Show users", "db_id": "mydb", "top_k": 5})

    assert "users" in result
    assert "orders" in result


def test_get_schema_tool_no_results() -> None:
    ctx = make_context()
    ctx.schema_retriever.retrieve.return_value = []  # type: ignore[attr-defined]

    tools = make_tools(ctx)
    get_schema = next(t for t in tools if t.name == "get_schema")
    result = get_schema.invoke({"question": "Show users", "db_id": "mydb", "top_k": 5})

    assert result == "No schema found."


def test_get_examples_tool_returns_formatted_text() -> None:
    ctx = make_context()
    docs = [
        ExampleDocument(db_id="mydb", question="How many users?", sql="SELECT COUNT(*) FROM users"),
    ]
    ctx.example_retriever.retrieve.return_value = docs  # type: ignore[attr-defined]

    tools = make_tools(ctx)
    get_examples = next(t for t in tools if t.name == "get_examples")
    result = get_examples.invoke({"question": "How many users?", "db_id": "mydb", "top_k": 3})

    assert "Q:" in result
    assert "SQL:" in result


def test_get_examples_tool_no_results() -> None:
    ctx = make_context()
    ctx.example_retriever.retrieve.return_value = []  # type: ignore[attr-defined]

    tools = make_tools(ctx)
    get_examples = next(t for t in tools if t.name == "get_examples")
    result = get_examples.invoke({"question": "How many users?", "db_id": "mydb", "top_k": 3})

    assert result == "No examples found."


def test_validate_sql_tool_valid() -> None:
    ctx = make_context()
    ctx.validator.validate.return_value = ValidationResult(valid=True)  # type: ignore[attr-defined]

    tools = make_tools(ctx)
    validate_sql = next(t for t in tools if t.name == "validate_sql")
    result = validate_sql.invoke({"sql": "SELECT * FROM users"})

    assert result == "valid"


def test_validate_sql_tool_invalid() -> None:
    ctx = make_context()
    ctx.validator.validate.return_value = ValidationResult(  # type: ignore[attr-defined]
        valid=False, error="syntax error near DROP"
    )

    tools = make_tools(ctx)
    validate_sql = next(t for t in tools if t.name == "validate_sql")
    result = validate_sql.invoke({"sql": "DROP TABLE users"})

    assert result.startswith("invalid:")


def test_execute_sql_tool_success() -> None:
    ctx = make_context()
    ctx.executor.execute.return_value = ExecutionResult(  # type: ignore[attr-defined]
        success=True,
        rows=[[1, "Alice"], [2, "Bob"]],
        column_names=["id", "name"],
        row_count=2,
    )

    tools = make_tools(ctx)
    execute_sql = next(t for t in tools if t.name == "execute_sql")
    result = execute_sql.invoke({"sql": "SELECT * FROM users", "db_id": "mydb"})

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert len(parsed["rows"]) == 2


def test_execute_sql_tool_failure() -> None:
    ctx = make_context()
    ctx.executor.execute.return_value = ExecutionResult(  # type: ignore[attr-defined]
        success=False,
        error="no such table: users",
        error_category="execution_error",
    )

    tools = make_tools(ctx)
    execute_sql = next(t for t in tools if t.name == "execute_sql")
    result = execute_sql.invoke({"sql": "SELECT * FROM users", "db_id": "mydb"})

    parsed = json.loads(result)
    assert parsed["success"] is False
    assert parsed["error"] is not None


def test_make_tools_returns_four_tools() -> None:
    ctx = make_context()
    tools = make_tools(ctx)
    assert len(tools) == 4


def test_tool_names() -> None:
    ctx = make_context()
    tools = make_tools(ctx)
    names = {t.name for t in tools}
    assert names == {"get_schema", "get_examples", "validate_sql", "execute_sql"}
