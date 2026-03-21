import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalExample:
    db_id: str
    question: str
    gold_sql: str


def load_dev_subset(
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
) -> list[EvalExample]:
    """Load evaluation examples from Spider dev.json with optional filtering."""
    dev_path = Path(spider_data_dir) / "dev.json"
    raw = json.loads(dev_path.read_text())
    examples = [
        EvalExample(db_id=e["db_id"], question=e["question"], gold_sql=e["query"])
        for e in raw
        if db_filter is None or e["db_id"] in db_filter
    ]
    return examples[:limit] if limit is not None else examples
