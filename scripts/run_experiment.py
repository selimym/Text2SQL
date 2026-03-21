#!/usr/bin/env python3
"""Experiment runner: compare pipeline variants on Spider subset."""

import argparse
import os
import sys
from typing import Literal, cast

import chromadb
import structlog

from app.core.config import get_settings
from app.core.llm import get_llm
from app.core.logging import configure_logging
from app.db.executor import SQLExecutor
from app.db.validator import SQLValidator
from app.eval.experiment import ComparisonReport, compare_variants
from app.llm.generator import SQLGenerator
from app.pipeline.assembler import PromptAssembler
from app.pipeline.factory import build_pipeline
from app.retrieval.embeddings import get_embeddings
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever

_log = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline variant comparison experiment")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "deterministic", "agent"],
        help="Pipeline variants to compare",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to evaluate")
    parser.add_argument(
        "--db-filter",
        nargs="*",
        default=None,
        help="Filter to specific database IDs",
    )
    parser.add_argument(
        "--output",
        default="experiment_report.json",
        help="Output file for ComparisonReport JSON",
    )
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="LangSmith project name (overrides LANGSMITH_PROJECT env var)",
    )
    return parser.parse_args()


def setup_langsmith(project: str | None) -> None:
    """Configure LangSmith tracing if LANGCHAIN_API_KEY is set."""
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        _log.info(
            "langsmith_tracing_enabled", project=os.environ.get("LANGCHAIN_PROJECT", "text2sql")
        )
    else:
        _log.info("langsmith_tracing_disabled", reason="LANGCHAIN_API_KEY not set")


def print_summary_table(report: ComparisonReport) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(
        f"{'Variant':<15} {'Exec Acc':<10} {'Exact Match':<12} {'Schema Recall':<14} {'Latency(ms)':<12} {'Retries':<8}"
    )
    print("-" * 80)
    for vr in report.variants:
        r = vr.report
        # avg retry_count across results if available, else N/A
        retries_list: list[int] = [
            int(row["retry_count"]) for row in r.results if row.get("retry_count") is not None
        ]
        avg_retries = sum(retries_list) / len(retries_list) if retries_list else None
        retries_str = f"{avg_retries:.1f}" if avg_retries is not None else "N/A"
        print(
            f"{vr.variant:<15} {r.execution_accuracy:<10.3f} {r.exact_match_rate:<12.3f} "
            f"{r.avg_schema_recall:<14.3f} {r.avg_latency_ms:<12.1f} {retries_str:<8}"
        )
    print("-" * 80)
    print(f"\nWinner: {report.winner}")
    print(f"Delta (execution_accuracy): {report.delta.get('execution_accuracy', 0):.3f}")
    print("=" * 80 + "\n")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    # Wire LangSmith
    langsmith_project = args.langsmith_project or settings.langsmith_project
    setup_langsmith(langsmith_project)

    # Build shared retrievers once
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embeddings = get_embeddings(settings.embedding_provider, settings.embedding_model)
    schema_retriever = SchemaRetriever(
        embeddings=embeddings,
        collection_name="spider_schemas",
        chroma_client=chroma_client,
    )
    example_retriever = ExampleRetriever(
        embeddings=embeddings,
        collection_name="spider_examples",
        chroma_client=chroma_client,
    )

    # Shared components
    llm = get_llm(settings.llm_provider, settings.llm_model)
    assembler = PromptAssembler()
    generator = SQLGenerator(llm=llm)
    validator = SQLValidator()
    executor = SQLExecutor(
        max_rows=settings.max_result_rows,
        timeout_seconds=settings.query_timeout_seconds,
    )

    # Build one pipeline per requested variant
    pipelines = {}
    for variant in args.variants:
        _log.info("building_pipeline", variant=variant)
        try:
            variant_literal = cast(Literal["baseline", "deterministic", "agent"], variant)
            pipelines[variant] = build_pipeline(
                schema_retriever=schema_retriever,
                example_retriever=example_retriever,
                assembler=assembler,
                generator=generator,
                validator=validator,
                executor=executor,
                spider_data_dir=settings.spider_data_dir,
                variant=variant_literal,
            )
        except NotImplementedError as e:
            _log.error("pipeline_variant_not_implemented", variant=variant, error=str(e))
            print(f"Error: {e}", file=sys.stderr)
            return 1

    _log.info(
        "starting_experiment", variants=args.variants, limit=args.limit, db_filter=args.db_filter
    )

    report = compare_variants(
        pipelines=pipelines,
        spider_data_dir=settings.spider_data_dir,
        db_filter=args.db_filter,
        limit=args.limit,
    )

    # Write JSON report
    with open(args.output, "w") as f:
        f.write(report.to_json())
    _log.info("report_saved", path=args.output)

    # Print summary table
    print_summary_table(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
