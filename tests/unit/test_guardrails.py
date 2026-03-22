from app.api.models import ExecutionMetadata, QueryResponse
from app.pipeline.guardrails import (
    check_input_guardrail,
    check_output_guardrail,
    check_retrieval_guardrail,
    check_sql_guardrail,
)

# --- check_input_guardrail ---


def test_input_empty_fails() -> None:
    result = check_input_guardrail("")
    assert result.passed is False


def test_input_short_fails() -> None:
    result = check_input_guardrail("hi")
    assert result.passed is False


def test_input_ddl_fails() -> None:
    result = check_input_guardrail("INSERT INTO users VALUES (1)")
    assert result.passed is False


def test_input_dml_delete_fails() -> None:
    result = check_input_guardrail("DELETE FROM users")
    assert result.passed is False


def test_input_too_long_fails() -> None:
    result = check_input_guardrail("a" * 2001)
    assert result.passed is False


def test_input_normal_passes() -> None:
    result = check_input_guardrail("How many singers are there?")
    assert result.passed is True
    assert result.reason is None


def test_input_drop_keyword_fails() -> None:
    result = check_input_guardrail("DROP the table called users")
    assert result.passed is False


# --- check_retrieval_guardrail ---


def test_retrieval_no_schema_fails() -> None:
    result = check_retrieval_guardrail([], ["some_example"])
    assert result.passed is False


def test_retrieval_with_schema_passes() -> None:
    result = check_retrieval_guardrail(["doc1", "doc2"], [])
    assert result.passed is True


def test_retrieval_truncation_warning() -> None:
    example_docs = ["e1", "e2", "e3", "e4", "e5", "e6"]
    result = check_retrieval_guardrail(["schema_doc"], example_docs, max_example_docs=5)
    assert result.passed is True
    assert result.reason is not None


# --- check_sql_guardrail ---


def test_sql_attach_fails() -> None:
    result = check_sql_guardrail("ATTACH DATABASE ':memory:' AS tmp")
    assert result.passed is False


def test_sql_pragma_fails() -> None:
    result = check_sql_guardrail("PRAGMA journal_mode")
    assert result.passed is False


def test_sql_sqlite_master_fails() -> None:
    result = check_sql_guardrail("SELECT * FROM sqlite_master")
    assert result.passed is False


def test_sql_no_limit_warns() -> None:
    result = check_sql_guardrail("SELECT * FROM singers")
    assert result.passed is True
    assert result.reason is not None


def test_sql_with_limit_passes() -> None:
    result = check_sql_guardrail("SELECT * FROM singers LIMIT 10")
    assert result.passed is True
    assert result.reason is None


def test_sql_count_no_limit_passes() -> None:
    result = check_sql_guardrail("SELECT COUNT(*) FROM singers")
    assert result.passed is True
    assert result.reason is None


# --- check_output_guardrail ---


def _make_response(success: bool | None) -> QueryResponse:
    metadata = None if success is None else ExecutionMetadata(success=success, row_count=1)
    return QueryResponse(
        question="test",
        generated_sql="SELECT 1",
        answer="1",
        execution_metadata=metadata,
    )


def test_output_success_passes() -> None:
    response = _make_response(True)
    result = check_output_guardrail(response)
    assert result.passed is True


def test_output_no_metadata_fails() -> None:
    response = _make_response(None)
    result = check_output_guardrail(response)
    assert result.passed is False


def test_output_execution_failed_fails() -> None:
    response = _make_response(False)
    result = check_output_guardrail(response)
    assert result.passed is False
