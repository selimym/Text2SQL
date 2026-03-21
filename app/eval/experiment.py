"""Multi-variant experiment comparison for Text2SQL evaluation."""

import dataclasses
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from app.eval.runner import EvalReport, run_evaluation
from app.pipeline.protocol import Pipeline


@dataclass
class VariantReport:
    variant: str
    report: EvalReport
    wall_clock_seconds: float

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return dataclasses.asdict(self)


@dataclass
class ComparisonReport:
    run_date: str
    limit: int | None
    db_filter: list[str] | None
    variants: list[VariantReport]
    winner: str
    delta: dict[str, float]

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compare_variants(
    pipelines: Mapping[str, Pipeline],
    spider_data_dir: str,
    db_filter: list[str] | None = None,
    limit: int | None = None,
) -> ComparisonReport:
    """Evaluate each pipeline variant and produce a comparison report."""
    variant_reports: list[VariantReport] = []

    for variant_name, pipeline in pipelines.items():
        start = time.monotonic()
        report = run_evaluation(pipeline, spider_data_dir, db_filter=db_filter, limit=limit)
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
        run_date=datetime.utcnow().isoformat(),
        limit=limit,
        db_filter=db_filter,
        variants=variant_reports,
        winner=winner,
        delta=delta,
    )
