"""DeterministicGraphPipeline: LangGraph-based Text2SQL pipeline with repair loop."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.api.models import QueryRequest, QueryResponse
from app.pipeline.nodes import NodeServices, bind_nodes
from app.pipeline.state import PipelineState


class DeterministicGraphPipeline:
    """LangGraph pipeline with 2-stage generation and conditional retry/repair loop.

    Topology::

        START → retrieve_schema → retrieve_examples → assemble_draft_prompt → generate_draft_sql
          generate_draft_sql → refine_schema_context → assemble_prompt → generate_final_sql → validate_sql
          validate_sql:
            - valid       → execute_sql
            - invalid, budget exhausted → build_response
            - invalid, retries remain   → critique_failure
          execute_sql:
            - success     → build_response
            - failure, budget exhausted → build_response
            - failure, retries remain   → critique_failure
          critique_failure:
            - retrieval_fault → broaden_schema → assemble_prompt
            - else            → assemble_prompt
          assemble_prompt → generate_final_sql  (repair loop)
          build_response → END
    """

    def __init__(self, services: NodeServices, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self._recursion_limit = (11 + 3) * (max_retries + 1)
        self._compiled: Any = self._build_graph(services, max_retries)

    def _build_graph(self, services: NodeServices, max_retries: int) -> Any:
        def route_after_validate(state: PipelineState) -> str:
            if state["validation_result"].valid:
                return "execute_sql"
            if state.get("retry_count", 0) >= max_retries:
                return "build_response"
            return "critique_failure"

        def route_after_execute(state: PipelineState) -> str:
            if state["execution_result"].success:
                return "build_response"
            if state.get("retry_count", 0) >= max_retries:
                return "build_response"
            return "critique_failure"

        def route_after_critique(state: PipelineState) -> str:
            if state.get("fault_category") == "retrieval_fault":
                return "broaden_schema"
            return "assemble_prompt"

        graph = StateGraph(PipelineState)
        bound = bind_nodes(services)
        for name, fn in bound.items():
            graph.add_node(name, fn)  # type: ignore[call-overload,unused-ignore]

        graph.add_edge(START, "retrieve_schema")
        graph.add_edge("retrieve_schema", "retrieve_examples")
        graph.add_edge("retrieve_examples", "assemble_draft_prompt")
        graph.add_edge("assemble_draft_prompt", "generate_draft_sql")
        graph.add_edge("generate_draft_sql", "refine_schema_context")
        graph.add_edge("refine_schema_context", "assemble_prompt")
        graph.add_edge("assemble_prompt", "generate_final_sql")
        graph.add_edge("generate_final_sql", "validate_sql")

        graph.add_conditional_edges(
            "validate_sql",
            route_after_validate,
            {
                "execute_sql": "execute_sql",
                "build_response": "build_response",
                "critique_failure": "critique_failure",
            },
        )

        graph.add_conditional_edges(
            "execute_sql",
            route_after_execute,
            {
                "build_response": "build_response",
                "critique_failure": "critique_failure",
            },
        )

        graph.add_conditional_edges(
            "critique_failure",
            route_after_critique,
            {
                "broaden_schema": "broaden_schema",
                "assemble_prompt": "assemble_prompt",
            },
        )

        graph.add_edge("broaden_schema", "assemble_prompt")
        graph.add_edge("build_response", END)

        return graph.compile()

    def run(self, request: QueryRequest) -> QueryResponse:
        initial_state: PipelineState = {"request": request}
        config = {"recursion_limit": self._recursion_limit}
        final_state = self._compiled.invoke(initial_state, config=config)

        exec_result = final_state.get("execution_result")
        gen_sql = final_state.get("generated_sql", "")
        draft_sql = final_state.get("draft_sql")
        retry_count = final_state.get("retry_count")
        step_timings = final_state.get("step_timings")

        flags: list[str] = []
        if retry_count is not None and retry_count > 0:
            flags.append(f"retried_{retry_count}_times")
        # Add max_retries_reached flag if we hit the limit
        validation = final_state.get("validation_result")
        if (validation and not validation.valid and (retry_count or 0) >= self.max_retries) or (
            exec_result and not exec_result.success and (retry_count or 0) >= self.max_retries
        ):
            flags.append("max_retries_reached")

        answer = ""
        if exec_result and exec_result.success:
            answer = str(exec_result.rows)
        elif exec_result:
            answer = f"Execution failed: {exec_result.error}"

        schema_docs = final_state.get("schema_docs", [])
        example_docs = final_state.get("example_docs", [])
        retrieved_example_sqls = [e.sql for e in example_docs]

        execution_metadata = None
        if exec_result:
            from app.api.models import ExecutionMetadata

            execution_metadata = ExecutionMetadata(
                success=exec_result.success,
                row_count=exec_result.row_count,
                latency_ms=exec_result.latency_ms,
                error=exec_result.error,
            )

        return QueryResponse(
            question=request.question,
            generated_sql=gen_sql,
            draft_sql=draft_sql,
            answer=answer,
            retrieved_schema_summary=[d.table_name for d in schema_docs],
            retrieved_examples_summary=[e.question for e in example_docs],
            retrieved_example_sqls=retrieved_example_sqls,
            execution_metadata=execution_metadata,
            step_timings=step_timings,
            retry_count=retry_count,
            flags=flags,
        )
