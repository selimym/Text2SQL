import sqlalchemy
from sqlalchemy import inspect as sa_inspect

from app.db.schema_doc import ColumnInfo, ForeignKeyInfo, SchemaDocument


def extract_schema(db_path: str) -> list[SchemaDocument]:
    """Extract schema information from a SQLite database file using SQLAlchemy inspection."""
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    inspector = sa_inspect(engine)
    docs: list[SchemaDocument] = []

    for table_name in inspector.get_table_names():
        columns: list[ColumnInfo] = []
        pk_cols = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))

        for col in inspector.get_columns(table_name):
            columns.append(
                ColumnInfo(
                    name=str(col["name"]),
                    data_type=str(col["type"]),
                    is_primary_key=col["name"] in pk_cols,
                    is_nullable=bool(col.get("nullable", True)),
                )
            )

        foreign_keys: list[ForeignKeyInfo] = []
        for fk in inspector.get_foreign_keys(table_name):
            for from_col, to_col in zip(
                fk.get("constrained_columns", []),
                fk.get("referred_columns", []),
                strict=False,
            ):
                foreign_keys.append(
                    ForeignKeyInfo(
                        from_column=str(from_col),
                        to_table=str(fk["referred_table"]),
                        to_column=str(to_col),
                    )
                )

        docs.append(
            SchemaDocument(
                db_id=db_path,
                table_name=table_name,
                columns=columns,
                foreign_keys=foreign_keys,
            )
        )

    engine.dispose()
    return docs
