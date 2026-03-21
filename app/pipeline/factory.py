from typing import Protocol, runtime_checkable

from app.api.models import QueryRequest, QueryResponse
from app.db.executor import SQLExecutor
from app.db.validator import SQLValidator
from app.llm.generator import SQLGenerator
from app.pipeline.assembler import PromptAssembler
from app.pipeline.baseline import BaselinePipeline
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


@runtime_checkable
class Pipeline(Protocol):
    def run(self, request: QueryRequest) -> QueryResponse: ...


def build_pipeline(
    schema_retriever: SchemaRetriever,
    example_retriever: ExampleRetriever,
    assembler: PromptAssembler,
    generator: SQLGenerator,
    validator: SQLValidator,
    executor: SQLExecutor,
    spider_data_dir: str,
    variant: str = "baseline",
) -> Pipeline:
    if variant == "baseline":
        return BaselinePipeline(
            schema_retriever=schema_retriever,
            example_retriever=example_retriever,
            assembler=assembler,
            generator=generator,
            validator=validator,
            executor=executor,
            spider_data_dir=spider_data_dir,
        )
    raise NotImplementedError(
        f"Pipeline variant not yet implemented: {variant}. Available: baseline, deterministic, agent"
    )
