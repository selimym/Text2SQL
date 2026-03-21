from app.db.schema_doc import SchemaDocument
from app.llm.prompts import build_user_prompt
from app.retrieval.example_doc import ExampleDocument


class PromptAssembler:
    def assemble(
        self,
        question: str,
        schema_docs: list[SchemaDocument],
        example_docs: list[ExampleDocument],
    ) -> str:
        return build_user_prompt(question, schema_docs, example_docs)
