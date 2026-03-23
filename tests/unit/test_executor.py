import pytest
import sqlalchemy

from app.db.executor import SQLExecutor


@pytest.fixture
def executor() -> SQLExecutor:
    return SQLExecutor(max_rows=100, timeout_seconds=30)


@pytest.fixture
def small_executor() -> SQLExecutor:
    return SQLExecutor(max_rows=2, timeout_seconds=30)


@pytest.fixture
def populated_db(sqlite_db_path: str) -> str:
    """Insert some rows into the test database."""
    engine = sqlalchemy.create_engine(f"sqlite:///{sqlite_db_path}")
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("INSERT INTO singer VALUES (1, 'Alice')"))
        conn.execute(sqlalchemy.text("INSERT INTO singer VALUES (2, 'Bob')"))
        conn.execute(sqlalchemy.text("INSERT INTO singer VALUES (3, 'Carol')"))
    engine.dispose()
    return sqlite_db_path


async def test_successful_select(executor: SQLExecutor, populated_db: str) -> None:
    result = await executor.execute("SELECT * FROM singer", populated_db)
    assert result.success is True
    assert result.row_count == 3
    assert len(result.rows) == 3
    assert result.column_names == ["singer_id", "name"]


async def test_row_cap_truncates(small_executor: SQLExecutor, populated_db: str) -> None:
    result = await small_executor.execute("SELECT * FROM singer", populated_db)
    assert result.success is True
    assert result.row_count == 2  # capped at max_rows=2
    assert result.error is not None  # error message about truncation


async def test_invalid_sql_returns_error(executor: SQLExecutor, sqlite_db_path: str) -> None:
    result = await executor.execute("SELECT * FROM nonexistent_table", sqlite_db_path)
    assert result.success is False
    assert result.error is not None
    assert result.error_category == "execution_error"


async def test_latency_recorded(executor: SQLExecutor, populated_db: str) -> None:
    result = await executor.execute("SELECT * FROM singer", populated_db)
    assert result.latency_ms > 0
