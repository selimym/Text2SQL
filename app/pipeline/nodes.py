import os
import time
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.db.executor import SQLExecutor
from app.db.validator import SQLValidator
from app.llm.generator import SQLGenerator
from app.llm.prompts import build_repair_prompt
from app.pipeline.assembler import PromptAssembler
from app.pipeline.state import PipelineState
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


@dataclass
class NodeServices:
    schema_retriever: SchemaRetriever
    example_retriever: ExampleRetriever
    assembler: PromptAssembler
    generator: SQLGenerator
    validator: SQLValidator
    executor: SQLExecutor
    spider_data_dir: str
    llm: BaseChatModel


def _merge_timings(state: PipelineState, key: str, elapsed_ms: float) -> dict[str, float]:
    existing = dict(state.get("step_timings") or {})
    existing[key] = elapsed_ms
    return existing


def _new_state(base: PipelineState, **updates: object) -> PipelineState:
    result: dict[str, object] = dict(base)
    result.update(updates)
    return result  # type: ignore[return-value]


def retrieve_schema_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    req = state["request"]
    schema_docs = services.schema_retriever.retrieve(req.question, req.db_id, req.top_k_schema)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        schema_docs=schema_docs,
        step_timings=_merge_timings(state, "retrieve_schema", elapsed_ms),
    )


def retrieve_examples_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    req = state["request"]
    example_docs = services.example_retriever.retrieve(req.question, req.db_id, req.top_k_examples)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        example_docs=example_docs,
        step_timings=_merge_timings(state, "retrieve_examples", elapsed_ms),
    )


def assemble_prompt_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    req = state["request"]
    schema_docs = state.get("schema_docs") or []
    example_docs = state.get("example_docs") or []
    critique_text = state.get("critique_text")

    if critique_text:
        prompt = build_repair_prompt(
            question=req.question,
            schema_docs=schema_docs,
            example_docs=example_docs,
            critique=critique_text,
        )
    else:
        prompt = services.assembler.assemble(req.question, schema_docs, example_docs)

    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        assembled_prompt=prompt,
        step_timings=_merge_timings(state, "assemble_prompt", elapsed_ms),
    )


def generate_sql_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    assembled_prompt = state.get("assembled_prompt") or ""
    generated_sql = services.generator.generate(assembled_prompt)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        generated_sql=generated_sql,
        step_timings=_merge_timings(state, "generate_sql", elapsed_ms),
    )


def validate_sql_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    generated_sql = state.get("generated_sql") or ""
    validation_result = services.validator.validate(generated_sql)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        validation_result=validation_result,
        step_timings=_merge_timings(state, "validate_sql", elapsed_ms),
    )


def execute_sql_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    db_id = state["request"].db_id
    db_path = os.path.join(services.spider_data_dir, db_id, f"{db_id}.sqlite")
    generated_sql = state.get("generated_sql") or ""
    execution_result = services.executor.execute(generated_sql, db_path)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        execution_result=execution_result,
        step_timings=_merge_timings(state, "execute_sql", elapsed_ms),
    )


def critique_failure_node(state: PipelineState, services: NodeServices) -> PipelineState:
    execution_result = state.get("execution_result")
    error_message = (execution_result.error if execution_result else None) or "unknown error"

    classification_prompt = (
        "Classify this SQL generation failure as either 'retrieval_fault' "
        "(wrong/missing schema context) or 'generation_fault' "
        "(correct schema but wrong SQL logic). "
        "Respond with only one of those two strings. "
        f"Error: {error_message}"
    )

    fault_category = "generation_fault"
    try:
        response = services.llm.invoke([HumanMessage(content=classification_prompt)])
        content = response.content
        raw = content.strip().lower() if isinstance(content, str) else ""
        fault_category = "retrieval_fault" if "retrieval_fault" in raw else "generation_fault"
    except Exception:
        fault_category = "generation_fault"

    critique_text = (
        f"The previous SQL attempt failed with: {error_message}. "
        f"Fault classification: {fault_category}. "
        "Please correct the SQL query."
    )

    current_retry = state.get("retry_count") or 0
    return _new_state(
        state,
        fault_category=fault_category,
        critique_text=critique_text,
        retry_count=current_retry + 1,
    )


def broaden_schema_node(state: PipelineState, services: NodeServices) -> PipelineState:
    req = state["request"]
    broader_top_k = req.top_k_schema + 3
    schema_docs = services.schema_retriever.retrieve(req.question, req.db_id, broader_top_k)
    return _new_state(state, schema_docs=schema_docs)


def build_response_node(state: PipelineState, services: NodeServices) -> PipelineState:
    return state
