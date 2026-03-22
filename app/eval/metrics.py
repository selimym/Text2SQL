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


def schema_precision(gold_sql: str, retrieved_tables: list[str]) -> float:
    """Fraction of retrieved tables that appear in gold SQL.
    Returns 0.0 if retrieved_tables is empty (avoid div-by-zero)."""
    if not retrieved_tables:
        return 0.0
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
        _log.debug("schema_precision: failed to parse gold_sql", error=str(exc))
    retrieved_set = {t.lower() for t in retrieved_tables}
    return len(gold_tables & retrieved_set) / len(retrieved_tables)


def schema_noise_ratio(gold_sql: str, retrieved_tables: list[str]) -> float:
    """1 - schema_precision. Returns 0.0 if retrieved_tables is empty."""
    if not retrieved_tables:
        return 0.0
    return 1.0 - schema_precision(gold_sql, retrieved_tables)


def fewshot_table_overlap(gold_sql: str, retrieved_sqls: list[str]) -> float:
    """Average per-example fraction of tables in each retrieved_sql that overlap with gold SQL tables.
    Uses same sqlglot table extraction as schema_recall.
    Returns 0.0 if retrieved_sqls is empty."""
    if not retrieved_sqls:
        return 0.0

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
        _log.debug("fewshot_table_overlap: failed to parse gold_sql", error=str(exc))

    overlaps: list[float] = []
    for retrieved_sql in retrieved_sqls:
        retrieved_tables: set[str] = set()
        try:
            for stmt in sqlglot.parse(retrieved_sql):
                if stmt is not None:
                    ret_cte_aliases: set[str] = set()
                    for cte in stmt.find_all(sqlglot.exp.CTE):
                        if cte.alias:
                            ret_cte_aliases.add(cte.alias.lower())
                    for tbl in stmt.find_all(sqlglot.exp.Table):
                        if tbl.name and tbl.name.lower() not in ret_cte_aliases:
                            retrieved_tables.add(tbl.name.lower())
        except Exception as exc:
            _log.debug(
                "fewshot_table_overlap: failed to parse retrieved_sql",
                error=str(exc),
            )

        if not retrieved_tables:
            overlap = 0.0
        elif not gold_tables:
            overlap = 1.0
        else:
            overlap = len(gold_tables & retrieved_tables) / len(retrieved_tables)
        overlaps.append(overlap)

    return sum(overlaps) / len(overlaps) if overlaps else 0.0
