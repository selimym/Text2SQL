"""Pipeline node functions for the LangGraph-based Text2SQL pipeline.

Each node has the signature ``(state: PipelineState, services: NodeServices) -> PipelineState``.
LangGraph only calls nodes with ``(state)`` or ``(state, config: RunnableConfig)``, so these
2-argument nodes MUST be registered via ``bind_nodes`` which wraps each one in a
``functools.partial`` that pre-fills the ``services`` argument.  Never add a node directly to
the graph without going through ``bind_nodes``.
"""

import functools
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import sqlglot
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


def assemble_draft_prompt_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    req = state["request"]
    schema_docs = state.get("schema_docs") or []
    example_docs = state.get("example_docs") or []
    prompt = services.assembler.assemble(req.question, schema_docs, example_docs)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        assembled_prompt=prompt,
        step_timings=_merge_timings(state, "assemble_draft_prompt_ms", elapsed_ms),
    )


def generate_draft_sql_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    assembled_prompt = state.get("assembled_prompt") or ""
    draft_sql = services.generator.generate(assembled_prompt)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        draft_sql=draft_sql,
        step_timings=_merge_timings(state, "generate_draft_sql_ms", elapsed_ms),
    )


def refine_schema_context_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    draft_sql = state.get("draft_sql") or ""
    schema_docs = list(state.get("schema_docs") or [])
    req = state["request"]

    try:
        mentioned_tables: set[str] = set()
        for stmt in sqlglot.parse(draft_sql):
            if stmt is not None:
                cte_aliases: set[str] = set()
                for cte in stmt.find_all(sqlglot.exp.CTE):
                    if cte.alias:
                        cte_aliases.add(cte.alias.lower())
                for tbl in stmt.find_all(sqlglot.exp.Table):
                    if tbl.name and tbl.name.lower() not in cte_aliases:
                        mentioned_tables.add(tbl.name.lower())

        if mentioned_tables:
            # Filter existing docs to only those whose table_name is mentioned
            filtered_docs = [
                doc for doc in schema_docs if doc.table_name.lower() in mentioned_tables
            ]

            # For any mentioned table not already in filtered_docs, try to fetch it
            covered = {doc.table_name.lower() for doc in filtered_docs}
            for table_name in mentioned_tables:
                if table_name not in covered:
                    fetched = services.schema_retriever.retrieve(req.question, req.db_id, top_k=1)
                    if fetched:
                        filtered_docs.append(fetched[0])
                        covered.add(table_name)

            schema_docs = filtered_docs
    except Exception:
        pass  # safe fallback: keep existing schema_docs unchanged

    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        schema_docs=schema_docs,
        step_timings=_merge_timings(state, "refine_schema_ms", elapsed_ms),
    )


def generate_final_sql_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    assembled_prompt = state.get("assembled_prompt") or ""
    generated_sql = services.generator.generate(assembled_prompt)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        generated_sql=generated_sql,
        step_timings=_merge_timings(state, "generate_final_sql_ms", elapsed_ms),
    )


# Backward-compatibility alias
generate_sql_node = generate_final_sql_node


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
    db_path = os.path.join(services.spider_data_dir, "database", db_id, f"{db_id}.sqlite")
    generated_sql = state.get("generated_sql") or ""
    execution_result = services.executor.execute(generated_sql, db_path)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        execution_result=execution_result,
        step_timings=_merge_timings(state, "execute_sql", elapsed_ms),
    )


def critique_failure_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    execution_result = state.get("execution_result")
    validation_result = state.get("validation_result")
    error_message = (
        (execution_result.error if execution_result else None)
        or (validation_result.error if validation_result else None)
        or "unknown error"
    )

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
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        fault_category=fault_category,
        critique_text=critique_text,
        retry_count=current_retry + 1,
        step_timings=_merge_timings(state, "critique_failure", elapsed_ms),
    )


def broaden_schema_node(state: PipelineState, services: NodeServices) -> PipelineState:
    start = time.monotonic()
    req = state["request"]
    broader_top_k = req.top_k_schema + 3
    schema_docs = services.schema_retriever.retrieve(req.question, req.db_id, broader_top_k)
    elapsed_ms = (time.monotonic() - start) * 1000
    return _new_state(
        state,
        schema_docs=schema_docs,
        step_timings=_merge_timings(state, "broaden_schema", elapsed_ms),
    )


def build_response_node(state: PipelineState, services: NodeServices) -> PipelineState:
    # Terminal node: pipeline extracts QueryResponse from state after graph completes.
    return state


def bind_nodes(services: NodeServices) -> dict[str, Callable[[PipelineState], PipelineState]]:
    """Return a mapping of node name to a bound callable suitable for LangGraph registration.

    LangGraph invokes nodes with ``(state)`` or ``(state, config)``.  Because every node in this
    module requires ``services``, each one must be wrapped with ``functools.partial`` before being
    added to the graph.  Using this helper is the only sanctioned way to register nodes; it makes
    it impossible to accidentally add an unbound 2-argument node directly to the graph.
    """
    node_fns: list[tuple[str, Callable[[PipelineState, NodeServices], PipelineState]]] = [
        ("retrieve_schema", retrieve_schema_node),
        ("retrieve_examples", retrieve_examples_node),
        ("assemble_prompt", assemble_prompt_node),
        ("assemble_draft_prompt", assemble_draft_prompt_node),
        ("generate_draft_sql", generate_draft_sql_node),
        ("refine_schema_context", refine_schema_context_node),
        ("generate_final_sql", generate_final_sql_node),
        ("generate_sql", generate_sql_node),
        ("validate_sql", validate_sql_node),
        ("execute_sql", execute_sql_node),
        ("critique_failure", critique_failure_node),
        ("broaden_schema", broaden_schema_node),
        ("build_response", build_response_node),
    ]
    return {
        name: functools.partial(lambda state, fn=fn, svc=services: fn(state, svc))
        for name, fn in node_fns
    }
