"""Tests for DeterministicGraphPipeline (TDD).

All tests use a real DeterministicGraphPipeline with mocked NodeServices so
that the actual LangGraph wiring is exercised.
"""

from unittest.mock import AsyncMock, MagicMock

from app.api.models import QueryRequest
from app.db.executor import ExecutionResult
from app.db.schema_doc import SchemaDocument
from app.db.validator import ValidationResult
from app.pipeline.nodes import NodeServices
from app.retrieval.example_doc import ExampleDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(**kwargs: object) -> QueryRequest:
    defaults: dict[str, object] = {"question": "How many singers?", "db_id": "concert_singer"}
    defaults.update(kwargs)
    return QueryRequest(**defaults)


def make_services(
    generated_sql: str = "SELECT 1",
    validation_valid: bool = True,
    exec_success: bool = True,
    critique_response: str = "generation_fault",
    exec_error: str | None = None,
) -> NodeServices:
    schema_doc = SchemaDocument(db_id="test_db", table_name="singers", columns=[], foreign_keys=[])
    example_doc = ExampleDocument(
        db_id="test_db", question="How many?", sql="SELECT COUNT(*) FROM t"
    )

    from app.db.executor import SQLExecutor
    from app.db.validator import SQLValidator
    from app.llm.generator import SQLGenerator
    from app.pipeline.assembler import PromptAssembler
    from app.retrieval.example_retriever import ExampleRetriever
    from app.retrieval.schema_retriever import SchemaRetriever

    schema_retriever = MagicMock(spec=SchemaRetriever)
    schema_retriever.retrieve = AsyncMock(return_value=[schema_doc])

    example_retriever = MagicMock(spec=ExampleRetriever)
    example_retriever.retrieve = AsyncMock(return_value=[example_doc])

    assembler = MagicMock(spec=PromptAssembler)
    assembler.assemble.return_value = "assembled prompt"

    llm = MagicMock()
    llm_response = MagicMock()
    llm_response.content = critique_response
    llm.ainvoke = AsyncMock(return_value=llm_response)

    generator = MagicMock(spec=SQLGenerator)
    generator.agenerate = AsyncMock(return_value=(generated_sql, {}))
    generator.llm = llm

    validator = MagicMock(spec=SQLValidator)
    validator.validate.return_value = ValidationResult(
        valid=validation_valid, error=None if validation_valid else "syntax error"
    )

    executor = MagicMock(spec=SQLExecutor)
    executor.execute = AsyncMock(
        return_value=ExecutionResult(
            success=exec_success,
            rows=[["1"]] if exec_success else [],
            row_count=1 if exec_success else 0,
            error=exec_error,
        )
    )

    return NodeServices(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
        llm=llm,
    )


def make_pipeline(services: NodeServices, max_retries: int = 2):  # type: ignore[no-untyped-def]
    from app.pipeline.graph_pipeline import DeterministicGraphPipeline

    return DeterministicGraphPipeline(services=services, max_retries=max_retries)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_happy_path() -> None:
    """Valid SQL, execution success → correct sql in response, no max_retries_reached flag."""
    services = make_services(generated_sql="SELECT COUNT(*) FROM singers")
    pipeline = make_pipeline(services)
    request = make_request()

    response = await pipeline.run(request)

    assert response.generated_sql == "SELECT COUNT(*) FROM singers"
    assert "max_retries_reached" not in response.flags
    assert response.execution_metadata is not None
    assert response.execution_metadata.success is True


async def test_validation_fail_generation_fault() -> None:
    """Validation fails first time, critique returns 'generation_fault', second attempt succeeds."""
    from app.db.executor import SQLExecutor
    from app.db.schema_doc import SchemaDocument
    from app.db.validator import SQLValidator
    from app.llm.generator import SQLGenerator
    from app.pipeline.assembler import PromptAssembler
    from app.retrieval.example_doc import ExampleDocument
    from app.retrieval.example_retriever import ExampleRetriever
    from app.retrieval.schema_retriever import SchemaRetriever

    schema_doc = SchemaDocument(db_id="test_db", table_name="singers", columns=[], foreign_keys=[])
    example_doc = ExampleDocument(
        db_id="test_db", question="How many?", sql="SELECT COUNT(*) FROM t"
    )

    schema_retriever = MagicMock(spec=SchemaRetriever)
    schema_retriever.retrieve = AsyncMock(return_value=[schema_doc])

    example_retriever = MagicMock(spec=ExampleRetriever)
    example_retriever.retrieve = AsyncMock(return_value=[example_doc])

    assembler = MagicMock(spec=PromptAssembler)
    assembler.assemble.return_value = "assembled prompt"

    llm = MagicMock()
    llm_response = MagicMock()
    llm_response.content = "generation_fault"
    llm.ainvoke = AsyncMock(return_value=llm_response)

    generator = MagicMock(spec=SQLGenerator)
    generator.agenerate = AsyncMock(return_value=("SELECT 1", {}))
    generator.llm = llm

    validator = MagicMock(spec=SQLValidator)
    # First call invalid, second call valid
    validator.validate.side_effect = [
        ValidationResult(valid=False, error="syntax error"),
        ValidationResult(valid=True),
    ]

    executor = MagicMock(spec=SQLExecutor)
    executor.execute = AsyncMock(
        return_value=ExecutionResult(success=True, rows=[["1"]], row_count=1)
    )

    services = NodeServices(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
        llm=llm,
    )
    pipeline = make_pipeline(services, max_retries=2)
    request = make_request()

    response = await pipeline.run(request)

    assert response.retry_count == 1
    assert "max_retries_reached" not in response.flags


async def test_validation_fail_retrieval_fault() -> None:
    """Validation fails with retrieval_fault → broaden_schema is called → second attempt succeeds."""
    from app.db.executor import SQLExecutor
    from app.db.schema_doc import SchemaDocument
    from app.db.validator import SQLValidator
    from app.llm.generator import SQLGenerator
    from app.pipeline.assembler import PromptAssembler
    from app.retrieval.example_doc import ExampleDocument
    from app.retrieval.example_retriever import ExampleRetriever
    from app.retrieval.schema_retriever import SchemaRetriever

    schema_doc = SchemaDocument(db_id="test_db", table_name="singers", columns=[], foreign_keys=[])
    example_doc = ExampleDocument(
        db_id="test_db", question="How many?", sql="SELECT COUNT(*) FROM t"
    )

    schema_retriever = MagicMock(spec=SchemaRetriever)
    # First retrieve call (normal) returns narrow schema; second (broaden) returns broader schema
    broader_schema_doc = SchemaDocument(
        db_id="test_db", table_name="concerts", columns=[], foreign_keys=[]
    )
    schema_retriever.retrieve = AsyncMock(
        side_effect=[
            [schema_doc],  # retrieve_schema initial
            [schema_doc, broader_schema_doc],  # broaden_schema call
        ]
    )

    example_retriever = MagicMock(spec=ExampleRetriever)
    example_retriever.retrieve = AsyncMock(return_value=[example_doc])

    assembler = MagicMock(spec=PromptAssembler)
    assembler.assemble.return_value = "assembled prompt"

    llm = MagicMock()
    llm_response = MagicMock()
    llm_response.content = "retrieval_fault"
    llm.ainvoke = AsyncMock(return_value=llm_response)

    generator = MagicMock(spec=SQLGenerator)
    generator.agenerate = AsyncMock(return_value=("SELECT 1", {}))
    generator.llm = llm

    validator = MagicMock(spec=SQLValidator)
    # First validation fails, second passes (after broaden+reassemble)
    validator.validate.side_effect = [
        ValidationResult(valid=False, error="wrong table"),
        ValidationResult(valid=True),
    ]

    executor = MagicMock(spec=SQLExecutor)
    executor.execute = AsyncMock(
        return_value=ExecutionResult(success=True, rows=[["1"]], row_count=1)
    )

    services = NodeServices(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
        llm=llm,
    )
    pipeline = make_pipeline(services, max_retries=2)
    request = make_request()

    response = await pipeline.run(request)

    # broaden_schema was called (second retrieve call happened)
    assert schema_retriever.retrieve.call_count == 2
    assert response.retry_count == 1


async def test_max_retries_reached_validation() -> None:
    """Validation always fails → after max_retries exhausted, max_retries_reached in flags."""
    services = make_services(validation_valid=False)
    # Override validator to always fail
    pipeline = make_pipeline(services, max_retries=1)
    request = make_request()

    response = await pipeline.run(request)

    assert "max_retries_reached" in response.flags


async def test_execution_fail_triggers_repair() -> None:
    """Execution fails first time → repair loop triggers → retry_count == 1 in response."""
    from app.db.executor import SQLExecutor
    from app.db.schema_doc import SchemaDocument
    from app.db.validator import SQLValidator
    from app.llm.generator import SQLGenerator
    from app.pipeline.assembler import PromptAssembler
    from app.retrieval.example_doc import ExampleDocument
    from app.retrieval.example_retriever import ExampleRetriever
    from app.retrieval.schema_retriever import SchemaRetriever

    schema_doc = SchemaDocument(db_id="test_db", table_name="singers", columns=[], foreign_keys=[])
    example_doc = ExampleDocument(
        db_id="test_db", question="How many?", sql="SELECT COUNT(*) FROM t"
    )

    schema_retriever = MagicMock(spec=SchemaRetriever)
    schema_retriever.retrieve = AsyncMock(return_value=[schema_doc])

    example_retriever = MagicMock(spec=ExampleRetriever)
    example_retriever.retrieve = AsyncMock(return_value=[example_doc])

    assembler = MagicMock(spec=PromptAssembler)
    assembler.assemble.return_value = "assembled prompt"

    llm = MagicMock()
    llm_response = MagicMock()
    llm_response.content = "generation_fault"
    llm.ainvoke = AsyncMock(return_value=llm_response)

    generator = MagicMock(spec=SQLGenerator)
    generator.agenerate = AsyncMock(return_value=("SELECT 1", {}))
    generator.llm = llm

    validator = MagicMock(spec=SQLValidator)
    validator.validate.return_value = ValidationResult(valid=True)

    executor = MagicMock(spec=SQLExecutor)
    # First call fails, second call succeeds
    executor.execute = AsyncMock(
        side_effect=[
            ExecutionResult(success=False, rows=[], row_count=0, error="no such table"),
            ExecutionResult(success=True, rows=[["1"]], row_count=1),
        ]
    )

    services = NodeServices(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
        llm=llm,
    )
    pipeline = make_pipeline(services, max_retries=2)
    request = make_request()

    response = await pipeline.run(request)

    assert response.retry_count == 1


async def test_step_timings_populated() -> None:
    """Final state has step_timings dict with timing keys."""
    services = make_services()
    pipeline = make_pipeline(services)
    request = make_request()

    response = await pipeline.run(request)

    assert response.step_timings is not None
    assert len(response.step_timings) > 0
    # Should have at least the core pipeline steps
    expected_keys = {
        "retrieve_schema",
        "retrieve_examples",
        "assemble_draft_prompt_ms",
        "generate_draft_sql_ms",
        "refine_schema_ms",
        "assemble_prompt",
        "generate_final_sql_ms",
        "validate_sql",
        "execute_sql",
    }
    for key in expected_keys:
        assert key in response.step_timings, f"Missing timing key: {key}"


async def test_two_stage_draft_sql_populated() -> None:
    """Happy path: response.draft_sql is populated after 2-stage generation."""
    services = make_services(generated_sql="SELECT COUNT(*) FROM singers")
    pipeline = make_pipeline(services)
    request = make_request()

    response = await pipeline.run(request)

    assert response.draft_sql is not None


async def test_repair_loop_uses_assemble_prompt_not_draft() -> None:
    """Repair loop goes through assemble_prompt, not assemble_draft_prompt.

    assemble_draft_prompt_node calls assembler.assemble once (stage-1).
    After a validation failure, the repair loop calls assemble_prompt_node which
    also calls assembler.assemble once more — but never revisits assemble_draft_prompt.
    So assembler.assemble.call_count should be exactly 2 total (1 draft + 1 repair).
    """
    from app.db.executor import SQLExecutor
    from app.db.schema_doc import SchemaDocument
    from app.db.validator import SQLValidator
    from app.llm.generator import SQLGenerator
    from app.pipeline.assembler import PromptAssembler
    from app.retrieval.example_doc import ExampleDocument
    from app.retrieval.example_retriever import ExampleRetriever
    from app.retrieval.schema_retriever import SchemaRetriever

    schema_doc = SchemaDocument(db_id="test_db", table_name="singers", columns=[], foreign_keys=[])
    example_doc = ExampleDocument(
        db_id="test_db", question="How many?", sql="SELECT COUNT(*) FROM t"
    )

    schema_retriever = MagicMock(spec=SchemaRetriever)
    schema_retriever.retrieve = AsyncMock(return_value=[schema_doc])

    example_retriever = MagicMock(spec=ExampleRetriever)
    example_retriever.retrieve = AsyncMock(return_value=[example_doc])

    assembler = MagicMock(spec=PromptAssembler)
    assembler.assemble.return_value = "assembled prompt"

    llm = MagicMock()
    llm_response = MagicMock()
    llm_response.content = "generation_fault"
    llm.ainvoke = AsyncMock(return_value=llm_response)

    generator = MagicMock(spec=SQLGenerator)
    generator.agenerate = AsyncMock(return_value=("SELECT 1", {}))
    generator.llm = llm

    validator = MagicMock(spec=SQLValidator)
    # First call invalid (triggers repair), second call valid
    validator.validate.side_effect = [
        ValidationResult(valid=False, error="syntax error"),
        ValidationResult(valid=True),
    ]

    executor = MagicMock(spec=SQLExecutor)
    executor.execute = AsyncMock(
        return_value=ExecutionResult(success=True, rows=[["1"]], row_count=1)
    )

    services = NodeServices(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
        llm=llm,
    )
    pipeline = make_pipeline(services, max_retries=2)
    request = make_request()

    await pipeline.run(request)

    # assembler.assemble is called twice in the happy-path of stage-1:
    #   1. assemble_draft_prompt_node (always calls assembler.assemble)
    #   2. assemble_prompt_node on the initial pass (no critique_text yet → calls assembler.assemble)
    # The repair loop re-enters assemble_prompt_node with critique_text set, which causes
    # build_repair_prompt to be used instead — so assembler.assemble is NOT called again.
    # Total: 2 calls to assembler.assemble (draft + initial final-prompt), not 3.
    assert assembler.assemble.call_count == 2


async def test_max_retries_reached_execution() -> None:
    """Execution always fails → after max_retries exhausted, max_retries_reached in flags."""
    from app.db.executor import SQLExecutor
    from app.db.schema_doc import SchemaDocument
    from app.db.validator import SQLValidator
    from app.llm.generator import SQLGenerator
    from app.pipeline.assembler import PromptAssembler
    from app.retrieval.example_doc import ExampleDocument
    from app.retrieval.example_retriever import ExampleRetriever
    from app.retrieval.schema_retriever import SchemaRetriever

    schema_doc = SchemaDocument(db_id="test_db", table_name="singers", columns=[], foreign_keys=[])
    example_doc = ExampleDocument(
        db_id="test_db", question="How many?", sql="SELECT COUNT(*) FROM t"
    )

    schema_retriever = MagicMock(spec=SchemaRetriever)
    schema_retriever.retrieve = AsyncMock(return_value=[schema_doc])

    example_retriever = MagicMock(spec=ExampleRetriever)
    example_retriever.retrieve = AsyncMock(return_value=[example_doc])

    assembler = MagicMock(spec=PromptAssembler)
    assembler.assemble.return_value = "assembled prompt"

    llm = MagicMock()
    llm_response = MagicMock()
    llm_response.content = "generation_fault"
    llm.ainvoke = AsyncMock(return_value=llm_response)

    generator = MagicMock(spec=SQLGenerator)
    generator.agenerate = AsyncMock(return_value=("SELECT bad", {}))
    generator.llm = llm

    validator = MagicMock(spec=SQLValidator)
    validator.validate.return_value = ValidationResult(valid=True)

    executor = MagicMock(spec=SQLExecutor)
    # All execution attempts fail — exhausts the retry budget (max_retries=1 → 2 total attempts)
    executor.execute = AsyncMock(
        return_value=ExecutionResult(success=False, rows=[], row_count=0, error="no such table")
    )

    services = NodeServices(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
        llm=llm,
    )
    pipeline = make_pipeline(services, max_retries=1)
    request = make_request()

    response = await pipeline.run(request)

    assert "max_retries_reached" in response.flags
