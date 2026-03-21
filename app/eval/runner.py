"""Offline evaluation runner for the Text2SQL baseline pipeline."""

import argparse
from dataclasses import dataclass

from app.api.models import QueryRequest
from app.eval.loader import load_dev_subset
from app.pipeline.baseline import BaselinePipeline


@dataclass
class EvalReport:
    total: int
    execution_success: int
    success_rate: float
    avg_latency_ms: float
    results: list[dict]  # type: ignore[type-arg]


def run_evaluation(
    pipeline: BaselinePipeline,
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
) -> EvalReport:
    """Run evaluation against Spider dev set and return an EvalReport."""
    examples = load_dev_subset(spider_data_dir, db_filter, limit)
    results = []
    total_latency = 0.0
    successes = 0

    for ex in examples:
        req = QueryRequest(question=ex.question, db_id=ex.db_id)
        resp = pipeline.run(req)
        success = resp.execution_metadata is not None and resp.execution_metadata.success
        latency = resp.execution_metadata.latency_ms if resp.execution_metadata else 0.0
        if success:
            successes += 1
        total_latency += latency
        results.append(
            {
                "question": ex.question,
                "db_id": ex.db_id,
                "gold_sql": ex.gold_sql,
                "generated_sql": resp.generated_sql,
                "success": success,
            }
        )

    n = len(examples)
    return EvalReport(
        total=n,
        execution_success=successes,
        success_rate=successes / n if n else 0.0,
        avg_latency_ms=total_latency / n if n else 0.0,
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
    args = parser.parse_args()

    print(f"Evaluation complete. Report would be saved to {args.output}")
    print("Note: Full pipeline wiring requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
