import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from app.api.models import ExecutionMetadata, QueryRequest, QueryResponse
from app.pipeline.agent_tools import ToolContext, make_tools

__all__ = ["AgentPipeline"]

_SYSTEM_PROMPT = """You are an expert Text-to-SQL agent. Follow these steps in order for every question:

1. Call get_schema(question, db_id) to retrieve the relevant tables and columns.
2. Call get_examples(question, db_id) to retrieve similar SQL examples.
3. Write a SQL SELECT query using ONLY the table and column names from the schema you retrieved.
4. Call validate_sql(sql) to check your query. If invalid, fix it and validate again.
5. When the query is valid, output it as your final answer.

Rules:
- Use ONLY table and column names that appear in the schema returned by get_schema. Never invent names.
- Your final answer must be ONLY the SQL query — no explanation, no markdown, no code fences.
- If the question cannot be answered with the available schema, output: SELECT NULL
"""


def _extract_tool_trace(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract a compact tool call trace from the agent message history."""
    trace = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append({"tool": tc["name"], "args": tc["args"]})
        elif isinstance(msg, ToolMessage) and trace:
            trace[-1]["result"] = str(msg.content)[:300]
    return trace


class AgentPipeline:
    def __init__(
        self,
        tool_context: ToolContext,
        llm: BaseChatModel,
        max_iterations: int = 10,
    ) -> None:
        self.tool_context = tool_context
        self.llm = llm
        self.max_iterations = max_iterations
        self._agent: Any = self._build_agent()

    def _build_agent(self) -> Any:
        tools = make_tools(self.tool_context)
        return create_react_agent(self.llm, tools, prompt=_SYSTEM_PROMPT)

    async def run(self, request: QueryRequest) -> QueryResponse:
        start_ms = time.monotonic() * 1000
        self.tool_context.retrieved_tables.clear()

        human_msg = f"Generate SQL for this question: {request.question}\nDatabase: {request.db_id}"
        config = {"recursion_limit": self.max_iterations * 3 + 5}

        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=human_msg)]},
                config=config,
            )
        except Exception as e:
            return QueryResponse(
                question=request.question,
                generated_sql="",
                answer="",
                flags=["agent_error", str(e)[:200]],
                retrieved_schema_summary=list(self.tool_context.retrieved_tables),
                step_timings={"total_ms": time.monotonic() * 1000 - start_ms},
                retry_count=0,
            )

        total_ms = time.monotonic() * 1000 - start_ms
        retrieved_schema_summary = list(self.tool_context.retrieved_tables)
        messages = result["messages"]
        tool_trace = _extract_tool_trace(messages)

        # Extract final SQL from last AIMessage without tool calls
        raw_sql = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                raw_sql = str(msg.content).strip()
                break

        # Count retry_count = number of LLM tool-call rounds
        retry_count = sum(1 for msg in messages if isinstance(msg, AIMessage) and msg.tool_calls)

        # Strip markdown fences
        raw_sql = re.sub(r"```(?:sql)?\s*", "", raw_sql, flags=re.IGNORECASE)
        raw_sql = re.sub(r"```\s*", "", raw_sql)
        raw_sql = raw_sql.strip()

        if not raw_sql:
            return QueryResponse(
                question=request.question,
                generated_sql="",
                answer="",
                flags=["no_sql_generated"],
                retrieved_schema_summary=retrieved_schema_summary,
                step_timings={"total_ms": total_ms, "tool_trace": tool_trace},
                retry_count=retry_count,
            )

        # Validate and execute once
        validation = self.tool_context.validator.validate(raw_sql)
        if not validation.valid:
            return QueryResponse(
                question=request.question,
                generated_sql=raw_sql,
                answer="",
                flags=["validation_failed", validation.error or ""],
                retrieved_schema_summary=retrieved_schema_summary,
                step_timings={"total_ms": total_ms, "tool_trace": tool_trace},
                retry_count=retry_count,
            )

        db_path = (
            f"{self.tool_context.spider_data_dir}/database/{request.db_id}/{request.db_id}.sqlite"
        )
        exec_result = await self.tool_context.executor.execute(raw_sql, db_path)

        answer = (
            str(exec_result.rows)
            if exec_result.success
            else f"Execution failed: {exec_result.error}"
        )

        return QueryResponse(
            question=request.question,
            generated_sql=raw_sql,
            answer=answer,
            retrieved_schema_summary=retrieved_schema_summary,
            execution_metadata=ExecutionMetadata(
                success=exec_result.success,
                row_count=exec_result.row_count,
                latency_ms=exec_result.latency_ms,
                error=exec_result.error,
            ),
            step_timings={"total_ms": total_ms, "tool_trace": tool_trace},
            retry_count=retry_count,
        )
