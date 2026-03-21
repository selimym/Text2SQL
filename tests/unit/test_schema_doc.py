from app.db.schema_doc import ColumnInfo, ForeignKeyInfo, SchemaDocument


def test_to_text_contains_table_name() -> None:
    doc = SchemaDocument(
        db_id="concert_singer",
        table_name="singer",
        columns=[
            ColumnInfo(name="singer_id", data_type="INTEGER", is_primary_key=True),
            ColumnInfo(name="name", data_type="TEXT", is_primary_key=False),
        ],
        foreign_keys=[],
    )
    text = doc.to_text()
    assert "singer" in text
    assert "singer_id" in text
    assert "name" in text
    assert "PK" in text


def test_to_text_contains_fk() -> None:
    doc = SchemaDocument(
        db_id="concert_singer",
        table_name="concert",
        columns=[
            ColumnInfo(name="concert_id", data_type="INTEGER", is_primary_key=True),
            ColumnInfo(name="stadium_id", data_type="INTEGER", is_primary_key=False),
        ],
        foreign_keys=[
            ForeignKeyInfo(
                from_column="stadium_id",
                to_table="stadium",
                to_column="stadium_id",
            )
        ],
    )
    text = doc.to_text()
    assert "FK" in text
    assert "stadium" in text


def test_to_chroma_metadata() -> None:
    doc = SchemaDocument(
        db_id="concert_singer",
        table_name="singer",
        columns=[],
        foreign_keys=[],
    )
    meta = doc.to_chroma_metadata()
    assert meta["db_id"] == "concert_singer"
    assert meta["table_name"] == "singer"
