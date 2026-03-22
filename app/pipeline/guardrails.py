from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.api.models import QueryResponse


@dataclass
class GuardrailResult:
    passed: bool
    reason: str | None = None


def check_input_guardrail(question: str) -> GuardrailResult:
    stripped = question.strip()
    if not stripped:
        return GuardrailResult(passed=False, reason="Question is empty or whitespace-only")
    if len(stripped) < 3:
        return GuardrailResult(passed=False, reason="Question is too short (minimum 3 characters)")
    if len(question) > 2000:
        return GuardrailResult(
            passed=False, reason="Question exceeds maximum length of 2000 characters"
        )
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b", question, re.IGNORECASE):
        return GuardrailResult(passed=False, reason="Question contains disallowed DDL/DML keywords")
    return GuardrailResult(passed=True)


def check_retrieval_guardrail(
    schema_docs: list[Any],
    example_docs: list[Any],
    max_schema_docs: int = 10,
    max_example_docs: int = 5,
) -> GuardrailResult:
    if len(schema_docs) == 0:
        return GuardrailResult(passed=False, reason="No schema documents retrieved")
    if len(schema_docs) > max_schema_docs:
        return GuardrailResult(
            passed=True,
            reason=f"schema_docs exceeds limit ({len(schema_docs)} > {max_schema_docs}); caller must truncate",
        )
    if len(example_docs) > max_example_docs:
        return GuardrailResult(
            passed=True,
            reason=f"example_docs exceeds limit ({len(example_docs)} > {max_example_docs}); caller must truncate",
        )
    return GuardrailResult(passed=True)


def check_sql_guardrail(sql: str) -> GuardrailResult:
    if re.search(r"\bATTACH\b", sql, re.IGNORECASE):
        return GuardrailResult(passed=False, reason="SQL contains disallowed ATTACH keyword")
    if re.search(r"\bPRAGMA\b", sql, re.IGNORECASE):
        return GuardrailResult(passed=False, reason="SQL contains disallowed PRAGMA keyword")
    if re.search(r"\bsqlite_master\b|\bsqlite_schema\b", sql, re.IGNORECASE):
        return GuardrailResult(
            passed=False, reason="SQL references disallowed sqlite internal tables"
        )
    has_limit = re.search(r"\bLIMIT\b", sql, re.IGNORECASE)
    has_aggregate = re.search(
        r"\bCOUNT\s*\(|\bSUM\s*\(|\bAVG\s*\(|\bMAX\s*\(|\bMIN\s*\(",
        sql,
        re.IGNORECASE,
    )
    if not has_limit and not has_aggregate:
        return GuardrailResult(
            passed=True, reason="Query has no LIMIT clause, may return many rows"
        )
    return GuardrailResult(passed=True)


def check_output_guardrail(response: QueryResponse) -> GuardrailResult:
    if response.execution_metadata is None or not response.execution_metadata.success:
        return GuardrailResult(passed=False, reason="Execution failed or produced no results")
    return GuardrailResult(passed=True)
