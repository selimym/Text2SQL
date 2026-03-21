"""Run EXPLAIN ANALYZE on SQL queries against PostgreSQL."""

from dataclasses import dataclass

import psycopg
from psycopg import sql as psql

from app.profiler.explain_parser import ExplainResult, parse_explain_output
from app.profiler.transpiler import sqlite_to_postgres


@dataclass
class ProfilingResult:
    sql: str
    transpiled_sql: str | None
    explain: ExplainResult
    error: str | None = None


def _empty_explain() -> ExplainResult:
    return ExplainResult(planning_time_ms=None, execution_time_ms=None, estimated_cost=None)


def run_explain_analyze(
    sql: str,
    postgres_url: str,
    search_path: str,
    transpile_from_sqlite: bool = True,
) -> ProfilingResult:
    """Run EXPLAIN ANALYZE on sql via the given Postgres connection.

    search_path is the Postgres schema name (e.g. 'spider_concert_singer').
    """
    transpiled: str | None = None
    try:
        transpiled = sqlite_to_postgres(sql) if transpile_from_sqlite else sql
    except Exception as e:
        return ProfilingResult(
            sql=sql,
            transpiled_sql=None,
            explain=_empty_explain(),
            error=f"Transpilation failed: {e}",
        )

    try:
        with psycopg.connect(postgres_url) as conn, conn.cursor() as cur:
            cur.execute(psql.SQL("SET search_path TO {}").format(psql.Identifier(search_path)))
            cur.execute(f"EXPLAIN ANALYZE {transpiled}")
            rows = cur.fetchall()
        explain_text = "\n".join(r[0] for r in rows)
        return ProfilingResult(
            sql=sql, transpiled_sql=transpiled, explain=parse_explain_output(explain_text)
        )
    except Exception as e:
        return ProfilingResult(
            sql=sql, transpiled_sql=transpiled, explain=_empty_explain(), error=str(e)
        )
