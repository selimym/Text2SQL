"""Load a Spider SQLite database into PostgreSQL via SQLAlchemy reflection."""

import argparse
from pathlib import Path

import sqlalchemy
from sqlalchemy import MetaData, Table, create_engine, text

from app.core.config import get_settings


def load_sqlite_to_postgres(sqlite_path: str, postgres_url: str, db_id: str) -> None:
    """Reflect SQLite schema + data into a Postgres schema named spider_{db_id}."""
    schema = f"spider_{db_id}"
    src_engine = create_engine(f"sqlite:///{sqlite_path}")
    dst_engine = create_engine(postgres_url)

    with dst_engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    src_meta = MetaData()
    with src_engine.connect() as conn:
        src_meta.reflect(bind=conn)

    dst_meta = MetaData(schema=schema)
    for table_name, src_table in src_meta.tables.items():
        Table(table_name, dst_meta, *[c.copy() for c in src_table.columns])
    with dst_engine.begin() as conn:
        dst_meta.create_all(bind=conn, checkfirst=True)

    with src_engine.connect() as src_conn, dst_engine.begin() as dst_conn:
        for table_name, _src_table in src_meta.tables.items():
            rows = src_conn.execute(sqlalchemy.text(f"SELECT * FROM {table_name}")).fetchall()
            if rows:
                dst_table = dst_meta.tables[f"{schema}.{table_name}"]
                dst_conn.execute(dst_table.insert(), [dict(r._mapping) for r in rows])

    n = len(src_meta.tables)
    print(f"Loaded {n} tables from {sqlite_path} into Postgres schema {schema}")
    src_engine.dispose()
    dst_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--spider-dir", default="spider_data")
    args = parser.parse_args()
    settings = get_settings()
    sqlite_path = str(Path(args.spider_dir) / "database" / args.db_id / f"{args.db_id}.sqlite")
    load_sqlite_to_postgres(sqlite_path, settings.postgres_profiler_url, args.db_id)


if __name__ == "__main__":
    main()
