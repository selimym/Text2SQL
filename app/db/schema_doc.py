from pydantic import BaseModel


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    is_primary_key: bool = False
    is_nullable: bool = True


class ForeignKeyInfo(BaseModel):
    from_column: str
    to_table: str
    to_column: str


class SchemaDocument(BaseModel):
    db_id: str
    table_name: str
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKeyInfo]

    def to_text(self) -> str:
        """Format schema as human-readable text for embedding."""
        parts = [f"Table: {self.table_name}"]
        col_parts = []
        for col in self.columns:
            desc = f"{col.name} ({col.data_type})"
            if col.is_primary_key:
                desc += " [PK]"
            col_parts.append(desc)
        parts.append("Columns: " + ", ".join(col_parts))
        for fk in self.foreign_keys:
            parts.append(f"FK: {fk.from_column} -> {fk.to_table}.{fk.to_column}")
        return "\n".join(parts)

    def to_chroma_metadata(self) -> dict[str, str]:
        """Return metadata dict for ChromaDB document storage."""
        return {"db_id": self.db_id, "table_name": self.table_name}
