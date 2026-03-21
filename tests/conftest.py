from pathlib import Path

import pytest
import sqlalchemy


@pytest.fixture
def sqlite_db_path(tmp_path: Path) -> str:
    """Create a temporary SQLite database with 2 tables and a FK for testing."""
    db_path = str(tmp_path / "test.db")
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("""
            CREATE TABLE singer (
                singer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        """)
        )
        conn.execute(
            sqlalchemy.text("""
            CREATE TABLE concert (
                concert_id INTEGER PRIMARY KEY,
                stadium_id INTEGER NOT NULL,
                FOREIGN KEY (stadium_id) REFERENCES singer(singer_id)
            )
        """)
        )
    return db_path
