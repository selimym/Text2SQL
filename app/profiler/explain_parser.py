"""Parser for PostgreSQL EXPLAIN ANALYZE text output."""

import re
from dataclasses import dataclass, field


@dataclass
class ExplainResult:
    planning_time_ms: float | None
    execution_time_ms: float | None
    estimated_cost: float | None
    node_types: list[str] = field(default_factory=list)


_NODE_PATTERNS: list[tuple[str, str]] = [
    (r"Index Only Scan", "IndexOnlyScan"),
    (r"Index Scan", "IndexScan"),
    (r"Bitmap Heap Scan", "BitmapHeapScan"),
    (r"Bitmap Index Scan", "BitmapIndexScan"),
    (r"Seq Scan", "SeqScan"),
    (r"Hash Join", "HashJoin"),
    (r"Nested Loop", "NestedLoop"),
    (r"Merge Join", "MergeJoin"),
    (r"Hash\b", "Hash"),
    (r"Sort\b", "Sort"),
    (r"Aggregate\b", "Aggregate"),
]


def parse_explain_output(text: str) -> ExplainResult:
    """Extract timing, cost, and node types from EXPLAIN ANALYZE output."""
    planning_time = None
    execution_time = None
    estimated_cost = None

    if m := re.search(r"Planning Time:\s+([\d.]+)\s+ms", text):
        planning_time = float(m.group(1))
    if m := re.search(r"Execution Time:\s+([\d.]+)\s+ms", text):
        execution_time = float(m.group(1))
    if m := re.search(r"cost=[\d.]+\.\.([\d.]+)", text):
        estimated_cost = float(m.group(1))

    node_types = [canonical for pattern, canonical in _NODE_PATTERNS if re.search(pattern, text)]
    return ExplainResult(
        planning_time_ms=planning_time,
        execution_time_ms=execution_time,
        estimated_cost=estimated_cost,
        node_types=node_types,
    )
