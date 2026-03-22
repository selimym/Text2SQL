import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy

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


@pytest.fixture
def spider_dir_with_db(tmp_path: Path) -> str:
    """spider_dir fixture that also creates a real SQLite DB for gold execution."""
    dev_data = [
        {
            "db_id": "concert_singer",
            "question": "How many singers?",
            "query": "SELECT COUNT(*) FROM singer",
        }
    ]
    (tmp_path / "dev.json").write_text(json.dumps(dev_data))
    db_dir = tmp_path / "database" / "concert_singer"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "concert_singer.sqlite"
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("CREATE TABLE singer (id INTEGER, name TEXT)"))
        conn.execute(sqlalchemy.text("INSERT INTO singer VALUES (1, 'Alice')"))
    engine.dispose()
    return str(tmp_path)


@pytest.fixture
def perfect_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.run.return_value = QueryResponse(
        question="How many singers?",
        generated_sql="SELECT COUNT(*) FROM singer",
        answer="[[1]]",
        retrieved_schema_summary=["singer"],
        execution_metadata=ExecutionMetadata(success=True, row_count=1, latency_ms=10.0),
    )
    return pipeline


def test_evalreport_has_new_fields(perfect_pipeline: MagicMock, spider_dir_with_db: str) -> None:
    report = run_evaluation(perfect_pipeline, spider_dir_with_db)
    assert hasattr(report, "execution_accuracy")
    assert hasattr(report, "exact_match_rate")
    assert hasattr(report, "avg_schema_recall")


def test_execution_accuracy_perfect(perfect_pipeline: MagicMock, spider_dir_with_db: str) -> None:
    report = run_evaluation(perfect_pipeline, spider_dir_with_db)
    assert report.execution_accuracy == 1.0


def test_exact_match_perfect(perfect_pipeline: MagicMock, spider_dir_with_db: str) -> None:
    report = run_evaluation(perfect_pipeline, spider_dir_with_db)
    assert report.exact_match_rate == 1.0


def test_schema_recall_perfect(perfect_pipeline: MagicMock, spider_dir_with_db: str) -> None:
    report = run_evaluation(perfect_pipeline, spider_dir_with_db)
    assert report.avg_schema_recall == 1.0


def test_results_contain_per_example_metrics(
    perfect_pipeline: MagicMock, spider_dir_with_db: str
) -> None:
    report = run_evaluation(perfect_pipeline, spider_dir_with_db)
    row = report.results[0]
    assert "execution_accuracy" in row
    assert "exact_match" in row
    assert "schema_recall" in row


def test_evalreport_has_new_precision_noise_fewshot_fields(
    perfect_pipeline: MagicMock, spider_dir_with_db: str
) -> None:
    report = run_evaluation(perfect_pipeline, spider_dir_with_db)
    assert hasattr(report, "avg_schema_precision")
    assert hasattr(report, "avg_schema_noise_ratio")
    assert hasattr(report, "avg_fewshot_table_overlap")


def test_avg_schema_precision_computed_from_retrieved_schema(spider_dir_with_db: str) -> None:
    pipeline = MagicMock()
    # retrieved_schema_summary contains 'singer', which is in gold_sql
    pipeline.run.return_value = QueryResponse(
        question="How many singers?",
        generated_sql="SELECT COUNT(*) FROM singer",
        answer="[[1]]",
        retrieved_schema_summary=["singer"],
        execution_metadata=ExecutionMetadata(success=True, row_count=1, latency_ms=10.0),
    )
    report = run_evaluation(pipeline, spider_dir_with_db)
    # precision = |{singer} & {singer}| / len([singer]) = 1/1 = 1.0
    assert report.avg_schema_precision == 1.0


def test_avg_fewshot_table_overlap_with_sqls(spider_dir_with_db: str) -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = QueryResponse(
        question="How many singers?",
        generated_sql="SELECT COUNT(*) FROM singer",
        answer="[[1]]",
        retrieved_schema_summary=["singer"],
        retrieved_example_sqls=["SELECT * FROM singer WHERE id = 1"],
        execution_metadata=ExecutionMetadata(success=True, row_count=1, latency_ms=10.0),
    )
    report = run_evaluation(pipeline, spider_dir_with_db)
    # fewshot overlap: gold tables={singer}, retrieved sql tables={singer} => 1.0
    assert report.avg_fewshot_table_overlap == 1.0


def test_avg_fewshot_table_overlap_none_when_no_sqls(spider_dir_with_db: str) -> None:
    pipeline = MagicMock()
    # retrieved_example_sqls is None (default)
    pipeline.run.return_value = QueryResponse(
        question="How many singers?",
        generated_sql="SELECT COUNT(*) FROM singer",
        answer="[[1]]",
        retrieved_schema_summary=["singer"],
        execution_metadata=ExecutionMetadata(success=True, row_count=1, latency_ms=10.0),
    )
    report = run_evaluation(pipeline, spider_dir_with_db)
    assert report.avg_fewshot_table_overlap == 0.0


def test_results_contain_new_per_example_metrics(spider_dir_with_db: str) -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = QueryResponse(
        question="How many singers?",
        generated_sql="SELECT COUNT(*) FROM singer",
        answer="[[1]]",
        retrieved_schema_summary=["singer"],
        retrieved_example_sqls=["SELECT name FROM singer"],
        execution_metadata=ExecutionMetadata(success=True, row_count=1, latency_ms=10.0),
    )
    report = run_evaluation(pipeline, spider_dir_with_db)
    row = report.results[0]
    assert "schema_precision" in row
    assert "schema_noise_ratio" in row
    assert "fewshot_table_overlap" in row
