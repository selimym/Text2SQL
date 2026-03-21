import json
import pathlib

import pytest

from app.db.spider_loader import load_schema_documents_from_spider

MINIMAL_TABLES_JSON = [
    {
        "db_id": "test_db",
        "table_names_original": ["singer", "concert"],
        "column_names_original": [
            [-1, "*"],
            [0, "singer_id"],
            [0, "name"],
            [1, "concert_id"],
            [1, "stadium_id"],
        ],
        "column_types": ["text", "number", "text", "number", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],
    }
]


@pytest.fixture
def tables_json_path(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "tables.json"
    path.write_text(json.dumps(MINIMAL_TABLES_JSON))
    return str(path)


def test_correct_doc_count(tables_json_path: str) -> None:
    docs = load_schema_documents_from_spider(tables_json_path)
    assert len(docs) == 2  # one per table


def test_primary_key_detected(tables_json_path: str) -> None:
    docs = load_schema_documents_from_spider(tables_json_path)
    singer_doc = next(d for d in docs if d.table_name == "singer")
    pk_cols = [c for c in singer_doc.columns if c.is_primary_key]
    assert len(pk_cols) == 1
    assert pk_cols[0].name == "singer_id"


def test_foreign_key_resolved(tables_json_path: str) -> None:
    docs = load_schema_documents_from_spider(tables_json_path)
    concert_doc = next(d for d in docs if d.table_name == "concert")
    assert len(concert_doc.foreign_keys) == 1
    fk = concert_doc.foreign_keys[0]
    assert fk.from_column == "stadium_id"
    assert fk.to_table == "singer"
    assert fk.to_column == "singer_id"
