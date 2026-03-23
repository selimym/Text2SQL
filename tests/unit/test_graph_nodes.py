import inspect
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from app.api.models import QueryRequest
from app.db.executor import ExecutionResult
from app.db.schema_doc import ColumnInfo, SchemaDocument
from app.db.validator import ValidationResult
from app.pipeline.nodes import NodeServices
from app.pipeline.state import PipelineState
from app.retrieval.example_doc import ExampleDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(**kwargs: object) -> QueryRequest:
    defaults: dict[str, object] = {"question": "How many singers?", "db_id": "concert_singer"}
    defaults.update(kwargs)
    return QueryRequest(**defaults)


def make_schema_doc(table_name: str = "singer") -> SchemaDocument:
    return SchemaDocument(
        db_id="concert_singer",
        table_name=table_name,
        columns=[ColumnInfo(name="id", data_type="INTEGER", is_primary_key=True)],
        foreign_keys=[],
    )


def make_example_doc() -> ExampleDocument:
    return ExampleDocument(
        db_id="concert_singer",
        question="Count singers?",
        sql="SELECT COUNT(*) FROM singer",
    )


def make_base_state(**kwargs: object) -> PipelineState:
    state: PipelineState = {"request": make_request()}
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state


def make_services(**overrides: object) -> NodeServices:
    """Return a NodeServices instance built from MagicMocks."""
    schema_retriever = MagicMock()
    schema_retriever.retrieve = AsyncMock(return_value=[make_schema_doc()])

    example_retriever = MagicMock()
    example_retriever.retrieve = AsyncMock(return_value=[make_example_doc()])

    assembler = MagicMock()
    assembler.assemble.return_value = "assembled prompt"

    generator = MagicMock()
    generator.agenerate = AsyncMock(return_value="SELECT COUNT(*) FROM singer")

    validator = MagicMock()
    validator.validate.return_value = ValidationResult(valid=True)

    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ExecutionResult(
            success=True, rows=[[1]], column_names=["count"], row_count=1, latency_ms=5.0
        )
    )

    llm = MagicMock()
    ai_message = MagicMock()
    ai_message.content = "generation_fault"
    llm.ainvoke = AsyncMock(return_value=ai_message)

    svc = NodeServices(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="/data/spider",
        llm=llm,
    )
    for key, val in overrides.items():
        setattr(svc, key, val)
    return svc


# ---------------------------------------------------------------------------
# retrieve_schema_node
# ---------------------------------------------------------------------------


async def test_retrieve_schema_node_updates_schema_docs() -> None:
    from app.pipeline.nodes import retrieve_schema_node

    state = make_base_state()
    svc = make_services()
    result = await retrieve_schema_node(state, svc)
    assert "schema_docs" in result
    assert len(result["schema_docs"]) == 1
    assert result["schema_docs"][0].table_name == "singer"


async def test_retrieve_schema_node_calls_retriever_with_correct_args() -> None:
    from app.pipeline.nodes import retrieve_schema_node

    state = make_base_state()
    svc = make_services()
    await retrieve_schema_node(state, svc)
    cast(AsyncMock, svc.schema_retriever.retrieve).assert_called_once_with(
        "How many singers?", "concert_singer", state["request"].top_k_schema
    )


async def test_retrieve_schema_node_updates_step_timings() -> None:
    from app.pipeline.nodes import retrieve_schema_node

    state = make_base_state()
    svc = make_services()
    result = await retrieve_schema_node(state, svc)
    assert "step_timings" in result
    assert "retrieve_schema" in result["step_timings"]
    assert result["step_timings"]["retrieve_schema"] >= 0.0


async def test_retrieve_schema_node_preserves_existing_timings() -> None:
    from app.pipeline.nodes import retrieve_schema_node

    state = make_base_state(step_timings={"previous_step": 12.5})
    svc = make_services()
    result = await retrieve_schema_node(state, svc)
    assert result["step_timings"]["previous_step"] == 12.5
    assert "retrieve_schema" in result["step_timings"]


# ---------------------------------------------------------------------------
# retrieve_examples_node
# ---------------------------------------------------------------------------


async def test_retrieve_examples_node_updates_example_docs() -> None:
    from app.pipeline.nodes import retrieve_examples_node

    state = make_base_state()
    svc = make_services()
    result = await retrieve_examples_node(state, svc)
    assert "example_docs" in result
    assert len(result["example_docs"]) == 1


async def test_retrieve_examples_node_updates_step_timings() -> None:
    from app.pipeline.nodes import retrieve_examples_node

    state = make_base_state()
    svc = make_services()
    result = await retrieve_examples_node(state, svc)
    assert "retrieve_examples" in result["step_timings"]


# ---------------------------------------------------------------------------
# assemble_prompt_node
# ---------------------------------------------------------------------------


async def test_assemble_prompt_node_without_critique_uses_normal_prompt() -> None:
    from app.pipeline.nodes import assemble_prompt_node

    state = make_base_state(
        schema_docs=[make_schema_doc()],
        example_docs=[make_example_doc()],
    )
    svc = make_services()
    result = await assemble_prompt_node(state, svc)
    assert result["assembled_prompt"] == "assembled prompt"
    cast(MagicMock, svc.assembler).assemble.assert_called_once()


async def test_assemble_prompt_node_with_critique_uses_repair_prompt() -> None:
    from app.pipeline.nodes import assemble_prompt_node

    state = make_base_state(
        schema_docs=[make_schema_doc()],
        example_docs=[make_example_doc()],
        critique_text="The table name is wrong.",
    )
    svc = make_services()
    with patch(
        "app.pipeline.nodes.build_repair_prompt", return_value="repair prompt"
    ) as mock_repair:
        result = await assemble_prompt_node(state, svc)
        mock_repair.assert_called_once_with(
            question=state["request"].question,
            schema_docs=state.get("schema_docs"),
            example_docs=state.get("example_docs"),
            critique=state.get("critique_text"),
        )
        assert result["assembled_prompt"] == "repair prompt"


async def test_assemble_prompt_node_updates_step_timings() -> None:
    from app.pipeline.nodes import assemble_prompt_node

    state = make_base_state(schema_docs=[make_schema_doc()], example_docs=[])
    svc = make_services()
    result = await assemble_prompt_node(state, svc)
    assert "assemble_prompt" in result["step_timings"]


# ---------------------------------------------------------------------------
# generate_final_sql_node
# ---------------------------------------------------------------------------


async def test_generate_final_sql_node_updates_generated_sql() -> None:
    from app.pipeline.nodes import generate_final_sql_node

    state = make_base_state(assembled_prompt="some prompt")
    svc = make_services()
    result = await generate_final_sql_node(state, svc)
    assert result["generated_sql"] == "SELECT COUNT(*) FROM singer"


async def test_generate_final_sql_node_updates_step_timings() -> None:
    from app.pipeline.nodes import generate_final_sql_node

    state = make_base_state(assembled_prompt="some prompt")
    svc = make_services()
    result = await generate_final_sql_node(state, svc)
    assert "generate_final_sql_ms" in result["step_timings"]


# ---------------------------------------------------------------------------
# validate_sql_node
# ---------------------------------------------------------------------------


async def test_validate_sql_node_updates_validation_result() -> None:
    from app.pipeline.nodes import validate_sql_node

    state = make_base_state(generated_sql="SELECT 1")
    svc = make_services()
    result = await validate_sql_node(state, svc)
    assert "validation_result" in result
    assert result["validation_result"].valid is True


async def test_validate_sql_node_updates_step_timings() -> None:
    from app.pipeline.nodes import validate_sql_node

    state = make_base_state(generated_sql="SELECT 1")
    svc = make_services()
    result = await validate_sql_node(state, svc)
    assert "validate_sql" in result["step_timings"]


# ---------------------------------------------------------------------------
# execute_sql_node
# ---------------------------------------------------------------------------


async def test_execute_sql_node_updates_execution_result() -> None:
    from app.pipeline.nodes import execute_sql_node

    state = make_base_state(generated_sql="SELECT COUNT(*) FROM singer")
    svc = make_services()
    result = await execute_sql_node(state, svc)
    assert "execution_result" in result
    assert result["execution_result"].success is True


async def test_execute_sql_node_builds_db_path_correctly() -> None:
    from app.pipeline.nodes import execute_sql_node

    state = make_base_state(generated_sql="SELECT 1")
    svc = make_services()
    await execute_sql_node(state, svc)
    # Should construct path from spider_data_dir + db_id + db_id.sqlite
    mock_executor = cast(AsyncMock, svc.executor.execute)
    mock_executor.assert_called_once()
    call_args = mock_executor.call_args
    db_path_arg = call_args[0][1] if call_args[0] else call_args[1]["db_path"]
    assert "concert_singer" in db_path_arg
    assert db_path_arg.endswith(".sqlite")


async def test_execute_sql_node_updates_step_timings() -> None:
    from app.pipeline.nodes import execute_sql_node

    state = make_base_state(generated_sql="SELECT 1")
    svc = make_services()
    result = await execute_sql_node(state, svc)
    assert "execute_sql" in result["step_timings"]


# ---------------------------------------------------------------------------
# critique_failure_node
# ---------------------------------------------------------------------------


async def test_critique_failure_node_llm_returns_retrieval_fault() -> None:
    from app.pipeline.nodes import critique_failure_node

    ai_message = MagicMock()
    ai_message.content = "retrieval_fault"
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=ai_message)

    state = make_base_state(
        generated_sql="SELECT x FROM missing_table",
        execution_result=ExecutionResult(
            success=False, error="no such table: missing_table", error_category="execution_error"
        ),
    )
    svc = make_services(llm=llm)
    result = await critique_failure_node(state, svc)
    assert result["fault_category"] == "retrieval_fault"
    assert result["critique_text"] is not None
    assert len(result["critique_text"]) > 0


async def test_critique_failure_node_llm_returns_generation_fault() -> None:
    from app.pipeline.nodes import critique_failure_node

    ai_message = MagicMock()
    ai_message.content = "generation_fault"
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=ai_message)

    state = make_base_state(
        generated_sql="SELECT wrong FROM singer",
        execution_result=ExecutionResult(
            success=False, error="no such column: wrong", error_category="execution_error"
        ),
    )
    svc = make_services(llm=llm)
    result = await critique_failure_node(state, svc)
    assert result["fault_category"] == "generation_fault"


async def test_critique_failure_node_llm_error_defaults_to_generation_fault() -> None:
    from app.pipeline.nodes import critique_failure_node

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    state = make_base_state(
        generated_sql="SELECT 1",
        execution_result=ExecutionResult(
            success=False, error="something broke", error_category="execution_error"
        ),
    )
    svc = make_services(llm=llm)
    result = await critique_failure_node(state, svc)
    assert result["fault_category"] == "generation_fault"


async def test_critique_failure_node_increments_retry_count() -> None:
    from app.pipeline.nodes import critique_failure_node

    state = make_base_state(
        generated_sql="SELECT 1",
        execution_result=ExecutionResult(success=False, error="err"),
        retry_count=1,
    )
    svc = make_services()
    result = await critique_failure_node(state, svc)
    assert result["retry_count"] == 2


async def test_critique_failure_node_increments_retry_count_from_zero() -> None:
    from app.pipeline.nodes import critique_failure_node

    state = make_base_state(
        generated_sql="SELECT 1",
        execution_result=ExecutionResult(success=False, error="err"),
    )
    svc = make_services()
    result = await critique_failure_node(state, svc)
    assert result["retry_count"] == 1


async def test_critique_failure_node_updates_step_timings() -> None:
    from app.pipeline.nodes import critique_failure_node

    state = make_base_state(
        generated_sql="SELECT 1",
        execution_result=ExecutionResult(success=False, error="err"),
    )
    svc = make_services()
    result = await critique_failure_node(state, svc)
    assert "critique_failure" in result["step_timings"]
    assert result["step_timings"]["critique_failure"] >= 0.0


# ---------------------------------------------------------------------------
# broaden_schema_node
# ---------------------------------------------------------------------------


async def test_broaden_schema_node_calls_retriever_with_broader_top_k() -> None:
    from app.pipeline.nodes import broaden_schema_node

    state = make_base_state(schema_docs=[make_schema_doc()])
    svc = make_services()
    await broaden_schema_node(state, svc)
    expected_top_k = state["request"].top_k_schema + 3
    cast(AsyncMock, svc.schema_retriever.retrieve).assert_called_once_with(
        state["request"].question, state["request"].db_id, expected_top_k
    )


async def test_broaden_schema_node_replaces_schema_docs() -> None:
    from app.pipeline.nodes import broaden_schema_node

    new_docs = [make_schema_doc("concert"), make_schema_doc("singer")]
    schema_retriever = MagicMock()
    schema_retriever.retrieve = AsyncMock(return_value=new_docs)
    state = make_base_state(schema_docs=[make_schema_doc("old_table")])
    svc = make_services(schema_retriever=schema_retriever)
    result = await broaden_schema_node(state, svc)
    assert result["schema_docs"] == new_docs


async def test_broaden_schema_node_updates_step_timings() -> None:
    from app.pipeline.nodes import broaden_schema_node

    state = make_base_state(schema_docs=[make_schema_doc()])
    svc = make_services()
    result = await broaden_schema_node(state, svc)
    assert "broaden_schema" in result["step_timings"]
    assert result["step_timings"]["broaden_schema"] >= 0.0


# ---------------------------------------------------------------------------
# build_response_node
# ---------------------------------------------------------------------------


async def test_build_response_node_returns_state_unchanged() -> None:
    from app.pipeline.nodes import build_response_node

    state = make_base_state(
        generated_sql="SELECT 1",
        execution_result=ExecutionResult(success=True, rows=[[1]], column_names=["c"], row_count=1),
    )
    svc = make_services()
    result = await build_response_node(state, svc)
    assert result["generated_sql"] == state["generated_sql"]
    assert result["request"] is state["request"]


# ---------------------------------------------------------------------------
# bind_nodes
# ---------------------------------------------------------------------------


def test_bind_nodes_returns_all_nodes() -> None:
    from app.pipeline.nodes import bind_nodes

    svc = make_services()
    bound = bind_nodes(svc)
    expected_names = {
        "retrieve_schema",
        "retrieve_examples",
        "assemble_prompt",
        "assemble_draft_prompt",
        "generate_draft_sql",
        "refine_schema_context",
        "generate_final_sql",
        "validate_sql",
        "execute_sql",
        "critique_failure",
        "broaden_schema",
        "build_response",
    }
    assert set(bound.keys()) == expected_names


def test_bind_nodes_callables_accept_only_state() -> None:
    from app.pipeline.nodes import bind_nodes

    svc = make_services()
    bound = bind_nodes(svc)
    for name, fn in bound.items():
        sig = inspect.signature(fn)
        # After binding, only `state` should remain as a required parameter
        free_params = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
        msg = f"Node '{name}' has {len(free_params)} free parameters after binding; expected 1"
        assert len(free_params) == 1, msg
