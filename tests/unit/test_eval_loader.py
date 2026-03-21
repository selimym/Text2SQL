import json
from pathlib import Path

import pytest

from app.eval.loader import load_dev_subset


@pytest.fixture
def spider_dir(tmp_path: Path) -> str:
    dev_data = [
        {
            "db_id": "concert_singer",
            "question": "How many singers?",
            "query": "SELECT COUNT(*) FROM singer",
        },
        {"db_id": "concert_singer", "question": "List concerts", "query": "SELECT * FROM concert"},
        {
            "db_id": "dog_kennels",
            "question": "How many dogs?",
            "query": "SELECT COUNT(*) FROM dogs",
        },
        {"db_id": "concert_singer", "question": "Find stadium", "query": "SELECT * FROM stadium"},
        {"db_id": "concert_singer", "question": "Singer names", "query": "SELECT name FROM singer"},
    ]
    dev_path = tmp_path / "dev.json"
    dev_path.write_text(json.dumps(dev_data))
    return str(tmp_path)


def test_load_all(spider_dir: str) -> None:
    examples = load_dev_subset(spider_dir)
    assert len(examples) == 5


def test_db_filter(spider_dir: str) -> None:
    examples = load_dev_subset(spider_dir, db_filter=["concert_singer"])
    assert len(examples) == 4
    assert all(e.db_id == "concert_singer" for e in examples)


def test_limit(spider_dir: str) -> None:
    examples = load_dev_subset(spider_dir, limit=2)
    assert len(examples) == 2


def test_db_filter_and_limit(spider_dir: str) -> None:
    examples = load_dev_subset(spider_dir, db_filter=["concert_singer"], limit=2)
    assert len(examples) == 2
    assert all(e.db_id == "concert_singer" for e in examples)


def test_eval_example_fields(spider_dir: str) -> None:
    examples = load_dev_subset(spider_dir)
    ex = examples[0]
    assert ex.db_id == "concert_singer"
    assert ex.question == "How many singers?"
    assert ex.gold_sql == "SELECT COUNT(*) FROM singer"
