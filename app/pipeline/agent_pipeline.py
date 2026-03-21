import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage as _HumanMessage  # noqa: F401 - keep for runtime
from langgraph.prebuilt import create_react_agent

from app.api.models import ExecutionMetadata, QueryRequest, QueryResponse
from app.pipeline.agent_tools import ToolContext, make_tools

__all__ = ["AgentPipeline"]


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
        system_prompt = (
            "You are an expert SQL generator. Use the available tools to retrieve schema "
            "information and examples, validate SQL, and execute queries. "
            "On your FINAL answer, respond with ONLY the SQL query — no explanation, no markdown, no code fences."
        )
        return create_react_agent(self.llm, tools, state_modifier=system_prompt)

    def run(self, request: QueryRequest) -> QueryResponse:
        from langchain_core.messages import HumanMessage

        start_ms = time.monotonic() * 1000

        human_msg = f"Generate SQL for this question: {request.question}\nDatabase: {request.db_id}"
        config = {"recursion_limit": self.max_iterations * 3}

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=human_msg)]},
            config=config,
        )

        total_ms = time.monotonic() * 1000 - start_ms

        # Extract final SQL from last AIMessage without tool calls
        messages = result["messages"]
        raw_sql = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                raw_sql = str(msg.content).strip()
                break

        # Strip markdown fences if present
        raw_sql = re.sub(r"^```(?:sql)?\s*", "", raw_sql, flags=re.IGNORECASE)
        raw_sql = re.sub(r"\s*```$", "", raw_sql)
        raw_sql = raw_sql.strip()

        # Count retry_count = number of tool-call rounds
        retry_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))

        # Validate and execute once
        validation = self.tool_context.validator.validate(raw_sql)
        if not validation.valid:
            return QueryResponse(
                question=request.question,
                generated_sql=raw_sql,
                answer="",
                flags=["validation_failed", validation.error or ""],
                step_timings={"total_ms": total_ms},
                retry_count=retry_count,
            )

        db_path = (
            f"{self.tool_context.spider_data_dir}/database/{request.db_id}/{request.db_id}.sqlite"
        )
        exec_result = self.tool_context.executor.execute(raw_sql, db_path)

        answer = (
            str(exec_result.rows)
            if exec_result.success
            else f"Execution failed: {exec_result.error}"
        )

        return QueryResponse(
            question=request.question,
            generated_sql=raw_sql,
            answer=answer,
            execution_metadata=ExecutionMetadata(
                success=exec_result.success,
                row_count=exec_result.row_count,
                latency_ms=exec_result.latency_ms,
                error=exec_result.error,
            ),
            step_timings={"total_ms": total_ms},
            retry_count=retry_count,
        )
