from app.db.introspection import extract_schema


def test_extract_schema_returns_correct_tables(sqlite_db_path: str) -> None:
    docs = extract_schema(sqlite_db_path)
    table_names = {d.table_name for d in docs}
    assert "singer" in table_names
    assert "concert" in table_names


def test_extract_schema_pk_detected(sqlite_db_path: str) -> None:
    docs = extract_schema(sqlite_db_path)
    singer_doc = next(d for d in docs if d.table_name == "singer")
    pk_cols = [c for c in singer_doc.columns if c.is_primary_key]
    assert len(pk_cols) == 1
    assert pk_cols[0].name == "singer_id"


def test_extract_schema_fk_detected(sqlite_db_path: str) -> None:
    docs = extract_schema(sqlite_db_path)
    concert_doc = next(d for d in docs if d.table_name == "concert")
    assert len(concert_doc.foreign_keys) == 1
    fk = concert_doc.foreign_keys[0]
    assert fk.from_column == "stadium_id"
    assert fk.to_table == "singer"
    assert fk.to_column == "singer_id"
