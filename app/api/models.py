import uuid

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    db_id: str
    top_k_schema: int = 5
    top_k_examples: int = 3


class ExecutionMetadata(BaseModel):
    success: bool
    row_count: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    answer: str
    retrieved_schema_summary: list[str] = Field(default_factory=list)
    retrieved_examples_summary: list[str] = Field(default_factory=list)
    execution_metadata: ExecutionMetadata | None = None
    confidence: float | None = None
    flags: list[str] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
