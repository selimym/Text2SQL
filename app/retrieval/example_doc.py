from pydantic import BaseModel


class ExampleDocument(BaseModel):
    db_id: str
    question: str
    sql: str
    tables_used: list[str] = []
    query_type: str | None = None
    difficulty: str | None = None

    def to_text(self) -> str:
        """Format example as text for embedding."""
        return f"Question: {self.question}\nSQL: {self.sql}"

    def to_chroma_metadata(self) -> dict[str, str]:
        """Return metadata dict for ChromaDB document storage."""
        return {
            "db_id": self.db_id,
            "question": self.question,
            "sql": self.sql,
        }
