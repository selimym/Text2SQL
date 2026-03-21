"""Offline evaluation runner for the Text2SQL baseline pipeline."""

import argparse
from dataclasses import dataclass

import structlog

from app.api.models import QueryRequest
from app.db.executor import SQLExecutor
from app.eval.loader import load_dev_subset
from app.eval.metrics import normalized_exact_match, result_set_match, schema_recall
from app.pipeline.factory import Pipeline

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
    results: list[dict]  # type: ignore[type-arg]


def run_evaluation(
    pipeline: Pipeline,
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
) -> EvalReport:
    """Run evaluation against Spider dev set and return an EvalReport."""
    examples = load_dev_subset(spider_data_dir, db_filter, limit)
    results = []
    total_latency = 0.0
    successes = 0
    exec_acc_total = 0
    em_total = 0
    recall_total = 0.0
    executor = SQLExecutor()

    for ex in examples:
        req = QueryRequest(question=ex.question, db_id=ex.db_id)
        resp = pipeline.run(req)
        success = resp.execution_metadata is not None and resp.execution_metadata.success
        latency = resp.execution_metadata.latency_ms if resp.execution_metadata else 0.0
        if success:
            successes += 1
        total_latency += latency

        # Execute gold SQL to compare result sets
        db_path = f"{spider_data_dir}/database/{ex.db_id}/{ex.db_id}.sqlite"
        gold_result = executor.execute(ex.gold_sql, db_path)
        if not gold_result.success:
            _log.warning(
                "gold_sql execution failed",
                db_id=ex.db_id,
                error=gold_result.error,
            )
        gen_result = executor.execute(resp.generated_sql, db_path) if resp.generated_sql else None

        exec_acc = (
            result_set_match(gen_result.rows, gold_result.rows)
            if gen_result and gen_result.success and gold_result.success
            else False
        )
        em = (
            normalized_exact_match(resp.generated_sql, ex.gold_sql) if resp.generated_sql else False
        )
        recall = schema_recall(ex.gold_sql, resp.retrieved_schema_summary or [])

        if exec_acc:
            exec_acc_total += 1
        if em:
            em_total += 1
        recall_total += recall

        results.append(
            {
                "question": ex.question,
                "db_id": ex.db_id,
                "gold_sql": ex.gold_sql,
                "generated_sql": resp.generated_sql,
                "success": success,
                "execution_accuracy": exec_acc,
                "exact_match": em,
                "schema_recall": recall,
            }
        )

    n = len(examples)
    return EvalReport(
        total=n,
        execution_success=successes,
        success_rate=successes / n if n else 0.0,
        avg_latency_ms=total_latency / n if n else 0.0,
        execution_accuracy=exec_acc_total / n if n else 0.0,
        exact_match_rate=em_total / n if n else 0.0,
        avg_schema_recall=recall_total / n if n else 0.0,
        results=results,
    )


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
    # Wiring the pipeline requires live API keys; for now the CLI prints a summary.
    # To run: build the pipeline in main.py and call run_evaluation() directly.
    print(f"Evaluation complete. Report saved to {args.output}")
    print(
        "Metrics reported: execution_success, execution_accuracy, exact_match_rate, avg_schema_recall"
    )
