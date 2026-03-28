import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.prompts import SYSTEM_PROMPT


class SQLGenerator:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def agenerate(self, user_prompt: str) -> str:
        """Async version of generate. Returns only the SQL string."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)
        sql = str(response.content).strip()
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)
        return sql.strip()

    def generate(self, user_prompt: str) -> tuple[str, dict[str, Any]]:
        """Generate SQL from a prompt. Returns (sql, usage) where usage may be empty."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        sql = str(response.content).strip()
        # Strip markdown code fences if present
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)
        usage: dict[str, Any] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = dict(response.usage_metadata)
        return sql.strip(), usage
