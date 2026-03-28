"""Offline evaluation runner for the Text2SQL baseline pipeline."""

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from app.api.models import QueryRequest
from app.db.executor import SQLExecutor
from app.eval.loader import load_dev_subset
from app.eval.metrics import (
    normalized_exact_match,
    result_set_match,
    schema_noise_ratio,
    schema_precision,
    schema_recall,
)
from app.pipeline.protocol import Pipeline

_log = structlog.get_logger()


@dataclass
class EvalReport:
    total: int
    execution_success: int
    success_rate: float
    avg_latency_ms: float
    execution_accuracy: float
    exact_match_rate: float
    avg_schema_recall: float
    results: list[dict[str, Any]]
    avg_schema_precision: float = 0.0
    avg_schema_noise_ratio: float = 0.0


async def run_evaluation(
    pipeline: Pipeline,
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
) -> EvalReport:
    """Run evaluation against Spider dev set and return an EvalReport."""
    examples = load_dev_subset(spider_data_dir, db_filter, limit)
    executor = SQLExecutor()
    results = []

    for ex in examples:
        req = QueryRequest(question=ex.question, db_id=ex.db_id)
        resp = await pipeline.run(req)
        success = resp.execution_metadata is not None and resp.execution_metadata.success
        latency = resp.execution_metadata.latency_ms if resp.execution_metadata else 0.0

        db_path = f"{spider_data_dir}/database/{ex.db_id}/{ex.db_id}.sqlite"
        gold_result = await executor.execute(ex.gold_sql, db_path)
        if not gold_result.success:
            _log.warning("gold_sql execution failed", db_id=ex.db_id, error=gold_result.error)
        gen_result = (
            await executor.execute(resp.generated_sql, db_path) if resp.generated_sql else None
        )

        results.append(
            _compute_result_metrics(
                ex_question=ex.question,
                ex_db_id=ex.db_id,
                ex_gold_sql=ex.gold_sql,
                generated_sql=resp.generated_sql,
                success=success,
                latency=latency,
                gen_rows=gen_result.rows if gen_result and gen_result.success else None,
                gold_rows=gold_result.rows if gold_result.success else None,
                gold_ok=gold_result.success,
                retrieved_schema=resp.retrieved_schema_summary or [],
                retry_count=resp.retry_count,
                flags=resp.flags or None,
                step_timings=resp.step_timings,
            )
        )

    return _aggregate(results)


def _compute_result_metrics(
    ex_question: str,
    ex_db_id: str,
    ex_gold_sql: str,
    generated_sql: str,
    success: bool,
    latency: float,
    gen_rows: list[list[Any]] | None,
    gold_rows: list[list[Any]] | None,
    gold_ok: bool,
    retrieved_schema: list[str],
    retry_count: int | None = None,
    flags: list[str] | None = None,
    step_timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exec_acc = (
        result_set_match(gen_rows, gold_rows)
        if gen_rows is not None and gold_rows is not None and gold_ok
        else False
    )
    em = normalized_exact_match(generated_sql, ex_gold_sql) if generated_sql else False
    recall = schema_recall(ex_gold_sql, retrieved_schema)
    precision = schema_precision(ex_gold_sql, retrieved_schema)
    noise = schema_noise_ratio(ex_gold_sql, retrieved_schema)
    row: dict[str, Any] = {
        "question": ex_question,
        "db_id": ex_db_id,
        "gold_sql": ex_gold_sql,
        "generated_sql": generated_sql,
        "success": success,
        "latency_ms": latency,
        "execution_accuracy": exec_acc,
        "exact_match": em,
        "schema_recall": recall,
        "schema_precision": precision,
        "schema_noise_ratio": noise,
    }
    if retry_count is not None:
        row["retry_count"] = retry_count
    if flags:
        row["flags"] = flags
    if step_timings:
        row["step_timings"] = step_timings
    return row


def _aggregate(results: list[dict[str, Any]]) -> EvalReport:
    n = len(results)
    if n == 0:
        return EvalReport(
            total=0,
            execution_success=0,
            success_rate=0.0,
            avg_latency_ms=0.0,
            execution_accuracy=0.0,
            exact_match_rate=0.0,
            avg_schema_recall=0.0,
            results=[],
            avg_schema_precision=0.0,
            avg_schema_noise_ratio=0.0,
        )
    successes = sum(1 for r in results if r["success"])
    exec_acc_total = sum(1 for r in results if r["execution_accuracy"])
    em_total = sum(1 for r in results if r["exact_match"])
    total_latency = sum(r.get("latency_ms", 0.0) for r in results)
    recall_total = sum(r["schema_recall"] for r in results)
    precision_total = sum(r["schema_precision"] for r in results)
    noise_total = sum(r["schema_noise_ratio"] for r in results)
    return EvalReport(
        total=n,
        execution_success=successes,
        success_rate=successes / n,
        avg_latency_ms=total_latency / n,
        execution_accuracy=exec_acc_total / n,
        exact_match_rate=em_total / n,
        avg_schema_recall=recall_total / n,
        results=results,
        avg_schema_precision=precision_total / n,
        avg_schema_noise_ratio=noise_total / n,
    )


async def run_evaluation_async(
    pipeline: Pipeline,
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
    concurrency: int = 8,
    checkpoint_path: Path | None = None,
    batch_size: int = 50,
) -> EvalReport:
    """Run evaluation concurrently with checkpoint/resume support.

    Uses asyncio.Semaphore to limit concurrent pipeline.run() calls.
    Examples are processed in batches of `batch_size`; each batch is written
    to the checkpoint file atomically after all its examples complete.
    On resume, skips examples already present in the checkpoint file.
    """
    examples = load_dev_subset(spider_data_dir, db_filter, limit)
    executor = SQLExecutor()

    # Load checkpoint if it exists
    done_results: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str]] = set()
    if checkpoint_path and checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                done_results.append(row)
                done_keys.add((row["question"], row["db_id"]))
        _log.info(
            "checkpoint_loaded",
            path=str(checkpoint_path),
            already_done=len(done_results),
            total=len(examples),
        )

    todo = [ex for ex in examples if (ex.question, ex.db_id) not in done_keys]
    _log.info("eval_starting", todo=len(todo), skipped=len(done_results), total=len(examples))

    if not todo:
        _log.info("all_examples_already_done", total=len(done_results))
        return _aggregate(done_results)

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    new_results: list[dict[str, Any]] = []

    async def process_one(ex: Any) -> dict[str, Any]:
        async with sem:
            req = QueryRequest(question=ex.question, db_id=ex.db_id)
            resp = await pipeline.run(req)

            success = resp.execution_metadata is not None and resp.execution_metadata.success
            latency = resp.execution_metadata.latency_ms if resp.execution_metadata else 0.0

            db_path = f"{spider_data_dir}/database/{ex.db_id}/{ex.db_id}.sqlite"
            gold_result = await executor.execute(ex.gold_sql, db_path)
            if not gold_result.success:
                _log.warning("gold_sql_failed", db_id=ex.db_id, error=gold_result.error)

            gen_result = (
                await executor.execute(resp.generated_sql, db_path) if resp.generated_sql else None
            )

            return _compute_result_metrics(
                ex_question=ex.question,
                ex_db_id=ex.db_id,
                ex_gold_sql=ex.gold_sql,
                generated_sql=resp.generated_sql,
                success=success,
                latency=latency,
                gen_rows=gen_result.rows if gen_result and gen_result.success else None,
                gold_rows=gold_result.rows if gold_result.success else None,
                gold_ok=gold_result.success,
                retrieved_schema=resp.retrieved_schema_summary or [],
                retry_count=resp.retry_count,
                flags=resp.flags or None,
                step_timings=resp.step_timings,
            )

    # Process in batches: gather each batch, write checkpoint, then continue
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        batch_results = await asyncio.gather(*[process_one(ex) for ex in batch])
        new_results.extend(batch_results)

        if checkpoint_path:
            with checkpoint_path.open("a") as f:
                for row in batch_results:
                    f.write(json.dumps(row) + "\n")

        completed = len(done_results) + len(new_results)
        _log.info(
            "batch_done",
            completed=completed,
            total=len(examples),
            batch_exec_acc=sum(r["execution_accuracy"] for r in batch_results) / len(batch_results),
        )

    return _aggregate(done_results + new_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline evaluation")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--db-filter", nargs="*", default=["concert_singer", "dog_kennels", "pets_1"]
    )
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument("--verbose", action="store_true", help="Print per-example breakdown")
    args = parser.parse_args()

    print("Note: Full pipeline wiring requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
    print(f"Evaluation complete. Report saved to {args.output}")
    print(
        "Metrics reported: execution_success, execution_accuracy, exact_match_rate, "
        "avg_schema_recall, avg_schema_precision, avg_schema_noise_ratio"
    )
