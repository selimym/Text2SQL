import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.prompts import SYSTEM_PROMPT


class SQLGenerator:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def generate(self, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        sql = str(response.content).strip()
        # Strip markdown code fences if present
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql)
        return sql.strip()
