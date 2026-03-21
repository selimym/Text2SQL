import json
from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from app.db.executor import SQLExecutor
from app.db.validator import SQLValidator
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


@dataclass
class ToolContext:
    schema_retriever: SchemaRetriever
    example_retriever: ExampleRetriever
    validator: SQLValidator
    executor: SQLExecutor
    spider_data_dir: str


class GetSchemaInput(BaseModel):
    question: str
    db_id: str
    top_k: int = 5


class GetExamplesInput(BaseModel):
    question: str
    db_id: str
    top_k: int = 3


class ValidateSqlInput(BaseModel):
    sql: str


class ExecuteSqlInput(BaseModel):
    sql: str
    db_id: str


def make_tools(context: ToolContext) -> list[BaseTool]:
    def _get_schema(question: str, db_id: str, top_k: int = 5) -> str:
        """Retrieve relevant database schema for a question."""
        docs = context.schema_retriever.retrieve(question, db_id, top_k)
        return "\n\n".join(doc.to_text() for doc in docs) if docs else "No schema found."

    def _get_examples(question: str, db_id: str, top_k: int = 3) -> str:
        """Retrieve similar SQL examples for a question."""
        docs = context.example_retriever.retrieve(question, db_id, top_k)
        if not docs:
            return "No examples found."
        return "\n\n".join(f"Q: {doc.question}\nSQL: {doc.sql}" for doc in docs)

    def _validate_sql(sql: str) -> str:
        """Validate SQL syntax."""
        result = context.validator.validate(sql)
        if result.valid:
            return "valid"
        return f"invalid: {result.error}"

    def _execute_sql(sql: str, db_id: str) -> str:
        """Execute SQL against the database."""
        db_path = f"{context.spider_data_dir}/database/{db_id}/{db_id}.sqlite"
        result = context.executor.execute(sql, db_path)
        return json.dumps(
            {
                "success": result.success,
                "rows": result.rows if result.success else [],
                "error": result.error,
            }
        )

    return [
        StructuredTool.from_function(
            func=_get_schema,
            name="get_schema",
            description="Retrieve relevant database schema for a question.",
            args_schema=GetSchemaInput,
        ),
        StructuredTool.from_function(
            func=_get_examples,
            name="get_examples",
            description="Retrieve similar SQL examples for a question.",
            args_schema=GetExamplesInput,
        ),
        StructuredTool.from_function(
            func=_validate_sql,
            name="validate_sql",
            description="Validate SQL syntax.",
            args_schema=ValidateSqlInput,
        ),
        StructuredTool.from_function(
            func=_execute_sql,
            name="execute_sql",
            description="Execute SQL against the database and return results.",
            args_schema=ExecuteSqlInput,
        ),
    ]
