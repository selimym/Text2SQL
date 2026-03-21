from typing import TypedDict

from app.api.models import QueryRequest
from app.db.executor import ExecutionResult
from app.db.schema_doc import SchemaDocument
from app.db.validator import ValidationResult
from app.retrieval.example_doc import ExampleDocument


class PipelineState(TypedDict, total=False):
    request: QueryRequest
    db_path: str
    schema_docs: list[SchemaDocument]
    example_docs: list[ExampleDocument]
    assembled_prompt: str
    generated_sql: str
    validation_result: ValidationResult
    execution_result: ExecutionResult
    retry_count: int
    fault_category: str | None
    critique_text: str | None
    step_timings: dict[str, float]
