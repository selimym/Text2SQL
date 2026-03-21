from unittest.mock import MagicMock

import pytest

from app.api.models import QueryRequest
from app.db.executor import ExecutionResult, SQLExecutor
from app.db.schema_doc import ColumnInfo, SchemaDocument
from app.db.validator import SQLValidator, ValidationResult
from app.llm.generator import SQLGenerator
from app.pipeline.assembler import PromptAssembler
from app.pipeline.baseline import BaselinePipeline
from app.pipeline.factory import Pipeline, build_pipeline
from app.pipeline.state import PipelineState
from app.retrieval.example_doc import ExampleDocument
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever

# ---------------------------------------------------------------------------
# PipelineState tests
# ---------------------------------------------------------------------------


def test_pipeline_state_empty() -> None:
    state: PipelineState = {}
    assert state == {}


def test_pipeline_state_partial_keys() -> None:
    state: PipelineState = {
        "generated_sql": "SELECT 1",
        "retry_count": 2,
    }
    assert state["generated_sql"] == "SELECT 1"
    assert state["retry_count"] == 2


def test_pipeline_state_all_keys() -> None:
    request = QueryRequest(question="How many singers?", db_id="concert_singer")
    schema_doc = SchemaDocument(
        db_id="concert_singer",
        table_name="singer",
        columns=[ColumnInfo(name="singer_id", data_type="INTEGER", is_primary_key=True)],
        foreign_keys=[],
    )
    example_doc = ExampleDocument(
        db_id="concert_singer",
        question="Count singers?",
        sql="SELECT COUNT(*) FROM singer",
    )
    validation_result = ValidationResult(valid=True)
    execution_result = ExecutionResult(
        success=True,
        rows=[[1]],
        column_names=["count"],
        row_count=1,
        latency_ms=5.0,
    )

    state: PipelineState = {
        "request": request,
        "db_path": "/data/concert_singer/concert_singer.sqlite",
        "schema_docs": [schema_doc],
        "example_docs": [example_doc],
        "assembled_prompt": "prompt text",
        "generated_sql": "SELECT COUNT(*) FROM singer",
        "validation_result": validation_result,
        "execution_result": execution_result,
        "retry_count": 0,
        "fault_category": None,
        "critique_text": None,
        "step_timings": {"retrieval": 12.5, "generation": 200.0},
    }

    assert state["request"] is request
    assert state["db_path"] == "/data/concert_singer/concert_singer.sqlite"
    assert state["schema_docs"] == [schema_doc]
    assert state["example_docs"] == [example_doc]
    assert state["assembled_prompt"] == "prompt text"
    assert state["generated_sql"] == "SELECT COUNT(*) FROM singer"
    assert state["validation_result"] is validation_result
    assert state["execution_result"] is execution_result
    assert state["retry_count"] == 0
    assert state["fault_category"] is None
    assert state["critique_text"] is None
    assert state["step_timings"] == {"retrieval": 12.5, "generation": 200.0}


# ---------------------------------------------------------------------------
# Pipeline Protocol + build_pipeline tests
# ---------------------------------------------------------------------------


def _make_mocked_services() -> (
    tuple[
        SchemaRetriever,
        ExampleRetriever,
        PromptAssembler,
        SQLGenerator,
        SQLValidator,
        SQLExecutor,
    ]
):
    schema_retriever = MagicMock(spec=SchemaRetriever)
    schema_retriever.retrieve.return_value = []
    example_retriever = MagicMock(spec=ExampleRetriever)
    example_retriever.retrieve.return_value = []
    assembler = MagicMock(spec=PromptAssembler)
    assembler.assemble.return_value = "assembled"
    generator = MagicMock(spec=SQLGenerator)
    generator.generate.return_value = "SELECT 1"
    validator = MagicMock(spec=SQLValidator)
    validator.validate.return_value = ValidationResult(valid=True)
    executor = MagicMock(spec=SQLExecutor)
    executor.execute.return_value = ExecutionResult(
        success=True, rows=[], column_names=[], row_count=0, latency_ms=1.0
    )
    return schema_retriever, example_retriever, assembler, generator, validator, executor


def test_build_pipeline_baseline_returns_baseline_pipeline() -> None:
    sr, er, asm, gen, val, exc = _make_mocked_services()
    pipeline = build_pipeline(
        schema_retriever=sr,
        example_retriever=er,
        assembler=asm,
        generator=gen,
        validator=val,
        executor=exc,
        spider_data_dir="/data/spider",
        variant="baseline",
    )
    assert isinstance(pipeline, BaselinePipeline)


def test_build_pipeline_baseline_default_variant() -> None:
    sr, er, asm, gen, val, exc = _make_mocked_services()
    pipeline = build_pipeline(
        schema_retriever=sr,
        example_retriever=er,
        assembler=asm,
        generator=gen,
        validator=val,
        executor=exc,
        spider_data_dir="/data/spider",
    )
    assert isinstance(pipeline, BaselinePipeline)


def test_baseline_pipeline_satisfies_protocol() -> None:
    sr, er, asm, gen, val, exc = _make_mocked_services()
    pipeline = build_pipeline(
        schema_retriever=sr,
        example_retriever=er,
        assembler=asm,
        generator=gen,
        validator=val,
        executor=exc,
        spider_data_dir="/data/spider",
        variant="baseline",
    )
    assert isinstance(pipeline, Pipeline)


def test_build_pipeline_unknown_variant_raises() -> None:
    sr, er, asm, gen, val, exc = _make_mocked_services()
    with pytest.raises(NotImplementedError, match="deterministic"):
        build_pipeline(
            schema_retriever=sr,
            example_retriever=er,
            assembler=asm,
            generator=gen,
            validator=val,
            executor=exc,
            spider_data_dir="/data/spider",
            variant="deterministic",
        )


def test_build_pipeline_unknown_variant_message_contains_available() -> None:
    sr, er, asm, gen, val, exc = _make_mocked_services()
    with pytest.raises(NotImplementedError, match="baseline, deterministic, agent"):
        build_pipeline(
            schema_retriever=sr,
            example_retriever=er,
            assembler=asm,
            generator=gen,
            validator=val,
            executor=exc,
            spider_data_dir="/data/spider",
            variant="agent",
        )
