from typing import cast
from unittest.mock import AsyncMock, MagicMock

from app.api.models import QueryRequest
from app.db.executor import ExecutionResult
from app.db.schema_doc import ColumnInfo, SchemaDocument
from app.db.validator import ValidationResult
from app.pipeline.baseline import BaselinePipeline
from app.retrieval.example_doc import ExampleDocument


def make_pipeline(
    sql: str = "SELECT COUNT(*) FROM singer",
    validation_valid: bool = True,
    exec_success: bool = True,
) -> BaselinePipeline:
    schema_retriever = MagicMock()
    schema_retriever.retrieve = AsyncMock(
        return_value=[
            SchemaDocument(
                db_id="concert_singer",
                table_name="singer",
                columns=[ColumnInfo(name="singer_id", data_type="INTEGER", is_primary_key=True)],
                foreign_keys=[],
            )
        ]
    )
    example_retriever = MagicMock()
    example_retriever.retrieve = AsyncMock(
        return_value=[
            ExampleDocument(
                db_id="concert_singer", question="Count singers?", sql="SELECT COUNT(*) FROM singer"
            )
        ]
    )
    assembler = MagicMock()
    assembler.assemble.return_value = "assembled prompt"
    generator = MagicMock()
    generator.agenerate = AsyncMock(return_value=sql)
    validator = MagicMock()
    validator.validate.return_value = ValidationResult(
        valid=validation_valid,
        error=None if validation_valid else "Only SELECT allowed",
    )
    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ExecutionResult(
            success=exec_success,
            rows=[[1]],
            column_names=["count"],
            row_count=1,
            latency_ms=10.0,
        )
    )
    return BaselinePipeline(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=assembler,
        generator=generator,
        validator=validator,
        executor=executor,
        spider_data_dir="spider_data",
    )


async def test_pipeline_full_flow() -> None:
    pipeline = make_pipeline()
    req = QueryRequest(question="How many singers?", db_id="concert_singer")
    resp = await pipeline.run(req)
    assert resp.generated_sql == "SELECT COUNT(*) FROM singer"
    assert resp.execution_metadata is not None
    assert resp.execution_metadata.success is True
    assert "singer" in resp.retrieved_schema_summary


async def test_pipeline_validation_failure() -> None:
    pipeline = make_pipeline(sql="DROP TABLE singer", validation_valid=False)
    req = QueryRequest(question="drop singers", db_id="concert_singer")
    resp = await pipeline.run(req)
    assert "validation_failed" in resp.flags
    assert resp.execution_metadata is None


async def test_pipeline_calls_schema_retriever() -> None:
    pipeline = make_pipeline()
    req = QueryRequest(question="How many singers?", db_id="concert_singer")
    await pipeline.run(req)
    cast(AsyncMock, pipeline.schema_retriever.retrieve).assert_called_once_with(
        "How many singers?", "concert_singer", req.top_k_schema
    )
