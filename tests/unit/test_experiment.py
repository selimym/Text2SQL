"""Tests for app/eval/experiment.py — compare_variants() and ComparisonReport."""

import json
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

from app.eval.experiment import compare_variants
from app.eval.runner import EvalReport
from app.pipeline.protocol import Pipeline


def make_eval_report(
    execution_accuracy: float = 0.7,
    exact_match: float = 0.3,
    schema_recall: float = 0.8,
) -> EvalReport:
    return EvalReport(
        total=10,
        execution_success=7,
        success_rate=0.7,
        avg_latency_ms=100.0,
        execution_accuracy=execution_accuracy,
        exact_match_rate=exact_match,
        avg_schema_recall=schema_recall,
        results=[],
    )


def _mock_pipeline() -> Pipeline:
    """Return a MagicMock that satisfies the Pipeline protocol."""
    p = MagicMock(spec=["run"])
    p.run = MagicMock()
    return p


# ---------------------------------------------------------------------------
# test 1 — winner is the variant with higher execution_accuracy
# ---------------------------------------------------------------------------


def test_compare_variants_winner_is_better_variant() -> None:
    report_a = make_eval_report(execution_accuracy=0.6)
    report_b = make_eval_report(execution_accuracy=0.8)

    pipelines: Mapping[str, Pipeline] = {
        "variant_a": _mock_pipeline(),
        "variant_b": _mock_pipeline(),
    }

    side_effects = [report_a, report_b]

    with patch("app.eval.experiment.run_evaluation", side_effect=side_effects):
        result = compare_variants(
            pipelines=pipelines,
            spider_data_dir="/fake/dir",
        )

    assert result.winner == "variant_b"


# ---------------------------------------------------------------------------
# test 2 — delta["execution_accuracy"] equals best - worst
# ---------------------------------------------------------------------------


def test_compare_variants_delta_execution_accuracy() -> None:
    report_a = make_eval_report(execution_accuracy=0.6)
    report_b = make_eval_report(execution_accuracy=0.8)

    pipelines: Mapping[str, Pipeline] = {
        "variant_a": _mock_pipeline(),
        "variant_b": _mock_pipeline(),
    }

    with patch("app.eval.experiment.run_evaluation", side_effect=[report_a, report_b]):
        result = compare_variants(
            pipelines=pipelines,
            spider_data_dir="/fake/dir",
        )

    assert abs(result.delta["execution_accuracy"] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# test 3 — JSON round-trip contains winner field
# ---------------------------------------------------------------------------


def test_comparison_report_json_serialization() -> None:
    report_a = make_eval_report(execution_accuracy=0.6)
    report_b = make_eval_report(execution_accuracy=0.8)

    pipelines: Mapping[str, Pipeline] = {
        "variant_a": _mock_pipeline(),
        "variant_b": _mock_pipeline(),
    }

    with patch("app.eval.experiment.run_evaluation", side_effect=[report_a, report_b]):
        result = compare_variants(
            pipelines=pipelines,
            spider_data_dir="/fake/dir",
        )

    json_str = result.to_json()
    parsed = json.loads(json_str)

    assert "winner" in parsed
    assert parsed["winner"] == "variant_b"


# ---------------------------------------------------------------------------
# test 4 — run_evaluation is called once per pipeline
# ---------------------------------------------------------------------------


def test_compare_variants_calls_run_evaluation_for_each_pipeline() -> None:
    report_a = make_eval_report(execution_accuracy=0.5)
    report_b = make_eval_report(execution_accuracy=0.7)
    report_c = make_eval_report(execution_accuracy=0.9)

    pipelines: Mapping[str, Pipeline] = {
        "alpha": _mock_pipeline(),
        "beta": _mock_pipeline(),
        "gamma": _mock_pipeline(),
    }

    with patch(
        "app.eval.experiment.run_evaluation", side_effect=[report_a, report_b, report_c]
    ) as mock_run:
        compare_variants(
            pipelines=pipelines,
            spider_data_dir="/fake/dir",
        )

    assert mock_run.call_count == 3


# ---------------------------------------------------------------------------
# test 5 — single variant: winner equals that variant's name
# ---------------------------------------------------------------------------


def test_compare_variants_single_variant() -> None:
    report = make_eval_report(execution_accuracy=0.75)

    pipelines: Mapping[str, Pipeline] = {"only_one": _mock_pipeline()}

    with patch("app.eval.experiment.run_evaluation", return_value=report):
        result = compare_variants(
            pipelines=pipelines,
            spider_data_dir="/fake/dir",
        )

    assert result.winner == "only_one"
