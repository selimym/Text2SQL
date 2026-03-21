"""Profile generated SQL queries using PostgreSQL EXPLAIN ANALYZE."""

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.profiler.pg_profiler import run_explain_analyze
from scripts.load_db_to_postgres import load_sqlite_to_postgres


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SQL via Postgres EXPLAIN ANALYZE")
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--spider-dir", default="spider_data")
    parser.add_argument(
        "--eval-results", help="Path to eval_report.json; uses generated_sql per row"
    )
    parser.add_argument("--sql", nargs="*", help="SQL queries to profile directly")
    parser.add_argument("--output", default="profiling_report.json")
    args = parser.parse_args()

    settings = get_settings()
    sqlite_path = str(Path(args.spider_dir) / "database" / args.db_id / f"{args.db_id}.sqlite")

    print(f"Loading {args.db_id} into Postgres...")
    load_sqlite_to_postgres(sqlite_path, settings.postgres_profiler_url, args.db_id)

    queries: list[str] = []
    if args.eval_results:
        data = json.loads(Path(args.eval_results).read_text())
        queries = [
            r["generated_sql"]
            for r in data.get("results", [])
            if r.get("db_id") == args.db_id and r.get("generated_sql")
        ]
    elif args.sql:
        queries = args.sql
    else:
        print("Provide --eval-results or --sql")
        return

    schema = f"spider_{args.db_id}"
    report = []
    for sql in queries:
        result = run_explain_analyze(
            sql=sql, postgres_url=settings.postgres_profiler_url, search_path=schema
        )
        report.append(
            {
                "sql": result.sql,
                "transpiled_sql": result.transpiled_sql,
                "error": result.error,
                "planning_time_ms": result.explain.planning_time_ms,
                "execution_time_ms": result.explain.execution_time_ms,
                "estimated_cost": result.explain.estimated_cost,
                "node_types": result.explain.node_types,
            }
        )

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Profiled {len(queries)} queries → {args.output}")


if __name__ == "__main__":
    main()
