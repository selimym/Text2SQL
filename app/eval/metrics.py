"""Pure metric functions for Text2SQL evaluation."""

import re
from collections import Counter
from typing import Any

import sqlglot
import structlog

_log = structlog.get_logger()


def result_set_match(
    actual: list[list[Any]],
    expected: list[list[Any]],
) -> bool:
    """Order-insensitive multiset comparison of two result sets."""

    def to_counts(rows: list[list[Any]]) -> Counter[tuple[Any, ...]]:
        return Counter(tuple(r) for r in rows)

    return to_counts(actual) == to_counts(expected)


def normalized_exact_match(generated: str, gold: str) -> bool:
    """Case-insensitive, whitespace-collapsed SQL string comparison."""

    def normalize(sql: str) -> str:
        return re.sub(r"\s+", " ", sql.strip().rstrip(";").strip().lower())

    return normalize(generated) == normalize(gold)


def schema_recall(gold_sql: str, retrieved_tables: list[str]) -> float:
    """Fraction of tables in gold_sql that appear in retrieved_tables."""
    gold_tables: set[str] = set()
    try:
        for stmt in sqlglot.parse(gold_sql):
            if stmt is not None:
                cte_aliases: set[str] = set()
                for cte in stmt.find_all(sqlglot.exp.CTE):
                    if cte.alias:
                        cte_aliases.add(cte.alias.lower())
                for tbl in stmt.find_all(sqlglot.exp.Table):
                    if tbl.name and tbl.name.lower() not in cte_aliases:
                        gold_tables.add(tbl.name.lower())
    except Exception as exc:
        _log.debug("schema_recall: failed to parse gold_sql", error=str(exc))
    if not gold_tables:
        return 1.0
    retrieved_set = {t.lower() for t in retrieved_tables}
    return len(gold_tables & retrieved_set) / len(gold_tables)
