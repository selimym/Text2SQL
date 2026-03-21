import pytest

from app.profiler.explain_parser import parse_explain_output
from app.profiler.transpiler import sqlite_to_postgres

SEQ_SCAN = (
    "Seq Scan on singer  (cost=0.00..1.01 rows=1 width=36)"
    " (actual time=0.012..0.013 rows=1 loops=1)\n"
    "Planning Time: 0.082 ms\n"
    "Execution Time: 0.034 ms"
)

INDEX_SCAN = (
    "Index Scan using singer_pkey on singer  (cost=0.00..8.02 rows=1 width=36)"
    " (actual time=0.018..0.019 rows=1 loops=1)\n"
    "Planning Time: 0.154 ms\n"
    "Execution Time: 0.031 ms"
)


def test_parses_planning_time() -> None:
    assert parse_explain_output(SEQ_SCAN).planning_time_ms == pytest.approx(0.082)


def test_parses_execution_time() -> None:
    assert parse_explain_output(SEQ_SCAN).execution_time_ms == pytest.approx(0.034)


def test_parses_estimated_cost() -> None:
    assert parse_explain_output(SEQ_SCAN).estimated_cost == pytest.approx(1.01)


def test_detects_seq_scan() -> None:
    assert "SeqScan" in parse_explain_output(SEQ_SCAN).node_types


def test_detects_index_scan() -> None:
    assert "IndexScan" in parse_explain_output(INDEX_SCAN).node_types


def test_missing_times_return_none() -> None:
    result = parse_explain_output("Seq Scan on t (cost=0.00..1.00 rows=1 width=4)")
    assert result.planning_time_ms is None
    assert result.execution_time_ms is None


def test_basic_select_unchanged() -> None:
    assert sqlite_to_postgres("SELECT * FROM singer") == "SELECT * FROM singer"


def test_backtick_identifiers_removed() -> None:
    result = sqlite_to_postgres("SELECT `name` FROM `singer`")
    assert "`" not in result


def test_returns_nonempty_string() -> None:
    result = sqlite_to_postgres("SELECT COUNT(*) FROM singer WHERE country = 'USA'")
    assert len(result) > 0
