"""Multi-variant experiment comparison for Text2SQL evaluation."""

import asyncio
import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.eval.runner import EvalReport, run_evaluation, run_evaluation_async
from app.pipeline.protocol import Pipeline


@dataclass
class VariantReport:
    variant: str
    report: EvalReport
    wall_clock_seconds: float

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class ComparisonReport:
    run_date: str
    limit: int | None
    db_filter: list[str] | None
    variants: list[VariantReport]
    winner: str
    delta: dict[str, float]
    similarity_threshold: float | None = None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


async def compare_variants(
    pipelines: Mapping[str, Pipeline],
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
    similarity_threshold: float | None = None,
) -> ComparisonReport:
    """Evaluate each pipeline variant and produce a comparison report."""
    if not pipelines:
        raise ValueError("compare_variants() requires at least one pipeline variant.")

    variant_reports: list[VariantReport] = []

    for variant_name, pipeline in pipelines.items():
        start = time.monotonic()
        report = await run_evaluation(pipeline, spider_data_dir, db_filter=db_filter, limit=limit)
        elapsed = time.monotonic() - start
        variant_reports.append(
            VariantReport(
                variant=variant_name,
                report=report,
                wall_clock_seconds=elapsed,
            )
        )

    winner = max(variant_reports, key=lambda vr: vr.report.execution_accuracy).variant

    metrics = ["execution_accuracy", "exact_match_rate", "avg_schema_recall"]
    delta: dict[str, float] = {}
    for metric in metrics:
        values = [getattr(vr.report, metric) for vr in variant_reports]
        delta[metric] = max(values) - min(values)

    return ComparisonReport(
        run_date=datetime.now(UTC).isoformat(),
        limit=limit,
        db_filter=db_filter,
        variants=variant_reports,
        winner=winner,
        delta=delta,
        similarity_threshold=similarity_threshold,
    )


def _run_id(
    limit: int | None, db_filter: list[str] | None, similarity_threshold: float | None
) -> str:
    key = f"{limit}_{sorted(db_filter) if db_filter else None}_{similarity_threshold}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


async def compare_variants_async(
    pipelines: Mapping[str, Pipeline],
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
    similarity_threshold: float | None = None,
    concurrency: int = 8,
    batch_size: int = 50,
    checkpoint_dir: Path | None = None,
) -> ComparisonReport:
    """Evaluate all pipeline variants in parallel, with concurrent example execution per variant.

    Each variant runs its examples concurrently up to `concurrency` (per-variant semaphore).
    All variants run simultaneously via asyncio.gather — total concurrent API calls is
    concurrency × number of variants.
    If checkpoint_dir is set, per-example results are saved to
    ``<checkpoint_dir>/<variant>_<run_id>.jsonl`` and resumed on re-run.
    """
    if not pipelines:
        raise ValueError("compare_variants_async() requires at least one pipeline variant.")

    run_id = _run_id(limit, db_filter, similarity_threshold)

    async def run_variant(variant_name: str, pipeline: Pipeline) -> VariantReport:
        ckpt_path = checkpoint_dir / f"{variant_name}_{run_id}.jsonl" if checkpoint_dir else None
        start = time.monotonic()
        report = await run_evaluation_async(
            pipeline,
            spider_data_dir,
            db_filter=db_filter,
            limit=limit,
            concurrency=concurrency,
            batch_size=batch_size,
            checkpoint_path=ckpt_path,
        )
        elapsed = time.monotonic() - start
        return VariantReport(variant=variant_name, report=report, wall_clock_seconds=elapsed)

    variant_reports: list[VariantReport] = await asyncio.gather(
        *[run_variant(name, pipeline) for name, pipeline in pipelines.items()]
    )

    winner = max(variant_reports, key=lambda vr: vr.report.execution_accuracy).variant

    metrics = ["execution_accuracy", "exact_match_rate", "avg_schema_recall"]
    delta: dict[str, float] = {}
    for metric in metrics:
        values = [getattr(vr.report, metric) for vr in variant_reports]
        delta[metric] = max(values) - min(values)

    return ComparisonReport(
        run_date=datetime.now(UTC).isoformat(),
        limit=limit,
        db_filter=db_filter,
        variants=list(variant_reports),
        winner=winner,
        delta=delta,
        similarity_threshold=similarity_threshold,
    )
