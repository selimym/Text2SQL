import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.api.models import ExecutionMetadata, QueryResponse
from app.eval.runner import run_evaluation


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.run.return_value = QueryResponse(
        question="How many singers?",
        generated_sql="SELECT COUNT(*) FROM singer",
        answer="[[3]]",
        execution_metadata=ExecutionMetadata(
            success=True,
            row_count=1,
            latency_ms=25.0,
        ),
    )
    return pipeline


@pytest.fixture
def spider_dir(tmp_path: Path) -> str:
    dev_data = [
        {
            "db_id": "concert_singer",
            "question": "How many singers?",
            "query": "SELECT COUNT(*) FROM singer",
        },
        {"db_id": "concert_singer", "question": "List concerts", "query": "SELECT * FROM concert"},
    ]
    (tmp_path / "dev.json").write_text(json.dumps(dev_data, ensure_ascii=False))
    return str(tmp_path)


def test_eval_report_fields(mock_pipeline: MagicMock, spider_dir: str) -> None:
    report = run_evaluation(mock_pipeline, spider_dir)
    assert report.total == 2
    assert report.execution_success == 2
    assert report.success_rate == 1.0
    assert report.avg_latency_ms == 25.0
    assert len(report.results) == 2


def test_eval_report_partial_success(spider_dir: str) -> None:
    pipeline = MagicMock()
    responses = [
        QueryResponse(
            question="q1",
            generated_sql="SELECT 1",
            answer="ok",
            execution_metadata=ExecutionMetadata(success=True, row_count=1, latency_ms=10.0),
        ),
        QueryResponse(
            question="q2",
            generated_sql="SELECT 2",
            answer="",
            execution_metadata=ExecutionMetadata(
                success=False, row_count=0, latency_ms=5.0, error="fail"
            ),
        ),
    ]
    pipeline.run.side_effect = responses
    report = run_evaluation(pipeline, spider_dir)
    assert report.total == 2
    assert report.execution_success == 1
    assert report.success_rate == 0.5
