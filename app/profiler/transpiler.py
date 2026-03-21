"""Dialect transpilation utilities using sqlglot."""

import sqlglot


def sqlite_to_postgres(sql: str) -> str:
    """Transpile a SQLite SQL statement to PostgreSQL dialect.

    Uses sqlglot. Raises on parse errors.
    """
    results = sqlglot.transpile(sql, read="sqlite", write="postgres")
    if not results:
        raise ValueError("sqlglot produced no output")
    return str(results[0])
