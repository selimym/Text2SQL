"""Pure metric functions for Text2SQL evaluation."""

from collections import Counter


def result_set_match(
    actual: list[list[object]],
    expected: list[list[object]],
) -> bool:
    """Order-insensitive multiset comparison of two result sets."""

    def to_counts(rows: list[list[object]]) -> Counter[tuple[object, ...]]:
        return Counter(tuple(r) for r in rows)

    return to_counts(actual) == to_counts(expected)
