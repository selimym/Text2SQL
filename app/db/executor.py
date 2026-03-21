import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

import sqlalchemy
from pydantic import BaseModel


class ExecutionResult(BaseModel):
    success: bool
    rows: list[list[object]] = []
    column_names: list[str] = []
    row_count: int = 0
    error: str | None = None
    error_category: str | None = None
    latency_ms: float = 0.0


class SQLExecutor:
    def __init__(self, max_rows: int = 100, timeout_seconds: int = 30) -> None:
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def execute(self, sql: str, db_path: str) -> ExecutionResult:
        start = time.monotonic()

        def _run() -> ExecutionResult:
            engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
            try:
                with engine.connect() as conn:
                    result = conn.execute(sqlalchemy.text(sql))
                    cols = list(result.keys())
                    rows = [list(r) for r in result.fetchmany(self.max_rows + 1)]
                    truncated = len(rows) > self.max_rows
                    rows = rows[: self.max_rows]
                    return ExecutionResult(
                        success=True,
                        rows=rows,
                        column_names=cols,
                        row_count=len(rows),
                        latency_ms=(time.monotonic() - start) * 1000,
                        error="Result truncated" if truncated else None,
                    )
            finally:
                engine.dispose()

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_run)
                return future.result(timeout=self.timeout_seconds)
        except FuturesTimeoutError:
            return ExecutionResult(
                success=False,
                error="Query timed out",
                error_category="timeout",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_category="execution_error",
                latency_ms=(time.monotonic() - start) * 1000,
            )
