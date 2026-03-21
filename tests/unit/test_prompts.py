from app.db.schema_doc import ColumnInfo, SchemaDocument
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.example_doc import ExampleDocument


def make_schema_doc(table_name: str) -> SchemaDocument:
    return SchemaDocument(
        db_id="test_db",
        table_name=table_name,
        columns=[ColumnInfo(name="id", data_type="INTEGER", is_primary_key=True)],
        foreign_keys=[],
    )


def make_example(question: str, sql: str) -> ExampleDocument:
    return ExampleDocument(db_id="test_db", question=question, sql=sql)


def test_prompt_contains_question() -> None:
    prompt = build_user_prompt(
        question="How many singers are there?",
        schema_docs=[make_schema_doc("singer")],
        example_docs=[],
    )
    assert "How many singers are there?" in prompt


def test_prompt_contains_table_name() -> None:
    prompt = build_user_prompt(
        question="test",
        schema_docs=[make_schema_doc("singer"), make_schema_doc("concert")],
        example_docs=[],
    )
    assert "singer" in prompt
    assert "concert" in prompt


def test_prompt_contains_example_sql() -> None:
    examples = [make_example("How many?", "SELECT COUNT(*) FROM singer")]
    prompt = build_user_prompt(
        question="test",
        schema_docs=[],
        example_docs=examples,
    )
    assert "SELECT COUNT(*) FROM singer" in prompt


def test_system_prompt_is_string() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
