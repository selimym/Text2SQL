from app.db.schema_doc import SchemaDocument
from app.retrieval.example_doc import ExampleDocument

SYSTEM_PROMPT = """You are an expert SQL generator. Given a natural language question, relevant database schema context, and similar examples, generate a single correct SQL SELECT query. Return ONLY the SQL query with no explanation or markdown."""


def build_user_prompt(
    question: str,
    schema_docs: list[SchemaDocument],
    example_docs: list[ExampleDocument],
) -> str:
    parts = ["## Schema Context"]
    for doc in schema_docs:
        parts.append(doc.to_text())
    if example_docs:
        parts.append("\n## Similar Examples")
        for ex in example_docs:
            parts.append(f"Q: {ex.question}\nSQL: {ex.sql}")
    parts.append(f"\n## Question\n{question}\n\nSQL:")
    return "\n\n".join(parts)


def build_repair_prompt(
    question: str,
    schema_docs: list[SchemaDocument],
    example_docs: list[ExampleDocument],
    critique: str,
) -> str:
    """Like build_user_prompt but prefixes with the critique context."""
    critique_section = (
        f"## Critique\n\n{critique}\n\nPlease fix the SQL based on the critique above."
    )
    main_prompt = build_user_prompt(question, schema_docs, example_docs)
    return critique_section + "\n\n" + main_prompt
