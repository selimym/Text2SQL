from typing import Literal

from app.db.executor import SQLExecutor
from app.db.validator import SQLValidator
from app.llm.generator import SQLGenerator
from app.pipeline.assembler import PromptAssembler
from app.pipeline.baseline import BaselinePipeline
from app.pipeline.protocol import Pipeline
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever

__all__ = ["Pipeline", "build_pipeline"]


def build_pipeline(
    schema_retriever: SchemaRetriever,
    example_retriever: ExampleRetriever,
    assembler: PromptAssembler,
    generator: SQLGenerator,
    validator: SQLValidator,
    executor: SQLExecutor,
    spider_data_dir: str,
    variant: Literal["baseline", "deterministic", "agent"] = "baseline",
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
    elif variant == "deterministic":
        from app.pipeline.graph_pipeline import DeterministicGraphPipeline
        from app.pipeline.nodes import NodeServices

        services = NodeServices(
            schema_retriever=schema_retriever,
            example_retriever=example_retriever,
            assembler=assembler,
            generator=generator,
            validator=validator,
            executor=executor,
            spider_data_dir=spider_data_dir,
            llm=generator.llm,
        )
        return DeterministicGraphPipeline(services=services, max_retries=2)
    elif variant == "agent":
        from app.pipeline.agent_pipeline import AgentPipeline
        from app.pipeline.agent_tools import ToolContext

        tool_context = ToolContext(
            schema_retriever=schema_retriever,
            example_retriever=example_retriever,
            validator=validator,
            executor=executor,
            spider_data_dir=spider_data_dir,
        )
        return AgentPipeline(
            tool_context=tool_context,
            llm=generator.llm,
            max_iterations=10,
        )
    raise NotImplementedError(
        f"Pipeline variant not yet implemented: {variant}. Available: baseline, deterministic, agent"
    )
