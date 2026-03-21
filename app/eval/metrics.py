"""Pure metric functions for Text2SQL evaluation."""

import re
from collections import Counter
from typing import Any


def result_set_match(
    actual: list[list[Any]],
    expected: list[list[Any]],
) -> bool:
    """Order-insensitive multiset comparison of two result sets."""

    def to_counts(rows: list[list[Any]]) -> Counter[tuple[Any, ...]]:
        return Counter(tuple(r) for r in rows)

    return to_counts(actual) == to_counts(expected)


def normalized_exact_match(generated: str, gold: str) -> bool:
    """Case-insensitive, whitespace-collapsed SQL string comparison."""

    def normalize(sql: str) -> str:
        return re.sub(r"\s+", " ", sql.strip().rstrip(";").strip().lower())

    return normalize(generated) == normalize(gold)
