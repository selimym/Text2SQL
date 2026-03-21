from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from app.profiler.explain_parser import parse_explain_output
from app.profiler.pg_profiler import ProfilingResult, run_explain_analyze
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


def _make_mock_conn(
    fetchall_return: Sequence[tuple[str, ...]] | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    mock_cur = MagicMock()
    if side_effect:
        mock_cur.fetchall.side_effect = side_effect
    else:
        mock_cur.fetchall.return_value = fetchall_return or []
    mock_cur.__enter__ = lambda s: mock_cur
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn


def test_run_explain_returns_profiling_result() -> None:
    rows = [
        (
            "Seq Scan on singer  (cost=0.00..1.01 rows=1 width=4) (actual time=0.010..0.011 rows=1 loops=1)",
        ),
        ("Planning Time: 0.080 ms",),
        ("Execution Time: 0.025 ms",),
    ]
    mock_conn = _make_mock_conn(fetchall_return=rows)
    with patch("app.profiler.pg_profiler.psycopg.connect", return_value=mock_conn):
        result = run_explain_analyze(
            sql="SELECT * FROM singer",
            postgres_url="postgresql://x:x@localhost/x",
            search_path="spider_concert_singer",
        )
    assert isinstance(result, ProfilingResult)
    assert result.explain.execution_time_ms == pytest.approx(0.025)
    assert result.error is None


def test_run_explain_handles_error() -> None:
    mock_conn = _make_mock_conn(side_effect=Exception("relation does not exist"))
    with patch("app.profiler.pg_profiler.psycopg.connect", return_value=mock_conn):
        result = run_explain_analyze(
            sql="SELECT * FROM nonexistent",
            postgres_url="postgresql://x:x@localhost/x",
            search_path="spider_concert_singer",
        )
    assert result.error is not None
    assert result.explain.execution_time_ms is None
