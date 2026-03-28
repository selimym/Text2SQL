"""Tests for the two-stage pipeline nodes: assemble_draft_prompt, generate_draft_sql,
refine_schema_context, and generate_final_sql."""

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
# Helpers (mirrors test_graph_nodes.py helpers)
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
    schema_retriever = MagicMock()
    schema_retriever.retrieve = AsyncMock(return_value=[make_schema_doc()])

    example_retriever = MagicMock()
    example_retriever.retrieve = AsyncMock(return_value=[make_example_doc()])

    assembler = MagicMock()
    assembler.assemble.return_value = "assembled prompt"

    generator = MagicMock()
    generator.agenerate = AsyncMock(return_value=("SELECT COUNT(*) FROM singer", {}))

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
# assemble_draft_prompt_node
# ---------------------------------------------------------------------------


async def test_assemble_draft_prompt_node_always_uses_assembler() -> None:
    """Even when critique_text is set, always calls assembler.assemble() and never build_repair_prompt."""
    from app.pipeline.nodes import assemble_draft_prompt_node

    state = make_base_state(
        schema_docs=[make_schema_doc()],
        example_docs=[make_example_doc()],
        critique_text="The table name is wrong.",
    )
    svc = make_services()
    with patch("app.pipeline.nodes.build_repair_prompt") as mock_repair:
        result = await assemble_draft_prompt_node(state, svc)
        mock_repair.assert_not_called()
    cast(MagicMock, svc.assembler).assemble.assert_called_once()
    assert result["assembled_prompt"] == "assembled prompt"


async def test_assemble_draft_prompt_node_updates_step_timings() -> None:
    from app.pipeline.nodes import assemble_draft_prompt_node

    state = make_base_state(schema_docs=[make_schema_doc()], example_docs=[])
    svc = make_services()
    result = await assemble_draft_prompt_node(state, svc)
    assert "step_timings" in result
    assert "assemble_draft_prompt_ms" in result["step_timings"]
    assert result["step_timings"]["assemble_draft_prompt_ms"] >= 0.0


# ---------------------------------------------------------------------------
# generate_draft_sql_node
# ---------------------------------------------------------------------------


async def test_generate_draft_sql_node_sets_draft_sql() -> None:
    """`draft_sql` is set in state; `generated_sql` is NOT set."""
    from app.pipeline.nodes import generate_draft_sql_node

    state = make_base_state(assembled_prompt="some prompt")
    svc = make_services()
    result = await generate_draft_sql_node(state, svc)
    assert result["draft_sql"] == "SELECT COUNT(*) FROM singer"
    assert "generated_sql" not in result


async def test_generate_draft_sql_node_updates_step_timings() -> None:
    from app.pipeline.nodes import generate_draft_sql_node

    state = make_base_state(assembled_prompt="some prompt")
    svc = make_services()
    result = await generate_draft_sql_node(state, svc)
    assert "generate_draft_sql_ms" in result["step_timings"]
    assert result["step_timings"]["generate_draft_sql_ms"] >= 0.0


# ---------------------------------------------------------------------------
# refine_schema_context_node
# ---------------------------------------------------------------------------


async def test_refine_schema_context_node_filters_to_mentioned_tables() -> None:
    """draft mentions table 'singer' → only singer doc kept."""
    from app.pipeline.nodes import refine_schema_context_node

    singer_doc = make_schema_doc("singer")
    concert_doc = make_schema_doc("concert")
    state = make_base_state(
        schema_docs=[singer_doc, concert_doc],
        draft_sql="SELECT COUNT(*) FROM singer",
    )
    svc = make_services()
    result = await refine_schema_context_node(state, svc)
    assert len(result["schema_docs"]) == 1
    assert result["schema_docs"][0].table_name == "singer"


async def test_refine_schema_context_node_keeps_all_when_parse_fails() -> None:
    """sqlglot parse failure → original schema_docs unchanged."""
    from app.pipeline.nodes import refine_schema_context_node

    singer_doc = make_schema_doc("singer")
    concert_doc = make_schema_doc("concert")
    original_docs = [singer_doc, concert_doc]

    state = make_base_state(
        schema_docs=original_docs,
        draft_sql="INVALID SQL !!!",
    )
    svc = make_services()

    with patch("app.pipeline.nodes.sqlglot.parse", side_effect=Exception("parse error")):
        result = await refine_schema_context_node(state, svc)

    assert len(result["schema_docs"]) == 2
    assert result["schema_docs"][0].table_name == "singer"
    assert result["schema_docs"][1].table_name == "concert"


async def test_refine_schema_context_node_all_mentioned_tables_have_docs() -> None:
    """If all mentioned tables already have docs, no retriever call is made."""
    from app.pipeline.nodes import refine_schema_context_node

    singer_doc = make_schema_doc("singer")
    state = make_base_state(
        schema_docs=[singer_doc],
        draft_sql="SELECT COUNT(*) FROM singer",
    )
    svc = make_services()
    result = await refine_schema_context_node(state, svc)
    cast(AsyncMock, svc.schema_retriever.retrieve).assert_not_called()
    assert len(result["schema_docs"]) == 1
    assert result["schema_docs"][0].table_name == "singer"


async def test_refine_schema_context_node_soft_fetch_missing_table() -> None:
    """A mentioned table not in schema_docs triggers a retriever call and appends whatever is returned."""
    from app.pipeline.nodes import refine_schema_context_node

    # draft SQL mentions 'concert', but only 'singer' doc exists in state
    singer_doc = make_schema_doc("singer")
    fetched_doc = make_schema_doc("venue")  # retriever returns a different table — still accepted

    state = make_base_state(
        schema_docs=[singer_doc],
        draft_sql="SELECT COUNT(*) FROM concert",
    )
    schema_retriever = MagicMock()
    schema_retriever.retrieve = AsyncMock(return_value=[fetched_doc])
    svc = make_services()
    svc.schema_retriever = schema_retriever

    result = await refine_schema_context_node(state, svc)

    schema_retriever.retrieve.assert_called_once()
    # singer_doc is filtered out (not mentioned); fetched_doc is appended
    assert len(result["schema_docs"]) == 1
    assert result["schema_docs"][0].table_name == "venue"


async def test_refine_schema_context_node_updates_step_timings() -> None:
    from app.pipeline.nodes import refine_schema_context_node

    state = make_base_state(
        schema_docs=[make_schema_doc()],
        draft_sql="SELECT COUNT(*) FROM singer",
    )
    svc = make_services()
    result = await refine_schema_context_node(state, svc)
    assert "refine_schema_ms" in result["step_timings"]
    assert result["step_timings"]["refine_schema_ms"] >= 0.0


# ---------------------------------------------------------------------------
# generate_final_sql_node
# ---------------------------------------------------------------------------


async def test_generate_final_sql_node_sets_generated_sql() -> None:
    """Sets `generated_sql`, NOT `draft_sql`."""
    from app.pipeline.nodes import generate_final_sql_node

    state = make_base_state(assembled_prompt="some prompt")
    svc = make_services()
    result = await generate_final_sql_node(state, svc)
    assert result["generated_sql"] == "SELECT COUNT(*) FROM singer"
    assert result.get("draft_sql") is None


async def test_generate_final_sql_node_updates_step_timings() -> None:
    from app.pipeline.nodes import generate_final_sql_node

    state = make_base_state(assembled_prompt="some prompt")
    svc = make_services()
    result = await generate_final_sql_node(state, svc)
    assert "generate_final_sql_ms" in result["step_timings"]
    assert result["step_timings"]["generate_final_sql_ms"] >= 0.0
