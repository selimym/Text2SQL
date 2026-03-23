# Design Decisions

This document records the *why* behind every major architectural choice in the Text2SQL backend. It is intended as a durable reference for contributors, reviewers, and future maintainers.

---

## 1. Why Dual RAG?

Schema retrieval answers "which tables and columns are relevant?" but tells the model nothing about *how* to write SQL for this database's dialect, join patterns, or aggregation idioms. Example retrieval fills that gap by surfacing similar past queries. Together they cover:

- **Schema retrieval** → what tables/columns to use
- **Example retrieval** → how to write the SQL (dialect, joins, subqueries, aggregation patterns)

Dropping either component degrades generation quality for complex, multi-table queries.

---

## 2. Why ChromaDB over Pinecone or Weaviate?

| Concern | ChromaDB advantage |
|---|---|
| External dependency | Runs fully in-process; no external service to stand up |
| Test isolation | `EphemeralClient` gives a throwaway in-memory store per test |
| Embedding API | Same `embed_documents` / `embed_query` interface as LangChain |
| Cost | Free, no API keys needed for the vector store itself |

Pinecone and Weaviate are better choices for multi-tenant production workloads, but add operational complexity that isn't justified at this scale.

---

## 3. Why provider-agnostic LLM/embeddings?

Factory functions (`get_llm`, `get_embeddings`) decouple cost/latency/quality tradeoffs from application code. Swapping Anthropic Claude for OpenAI GPT-4o, or `text-embedding-3-small` for a local sentence-transformer, requires only a single environment variable change — not a code change. This also makes it straightforward to run evals comparing providers.

---

## 4. Why three pipeline variants?

| Variant | Purpose |
|---|---|
| `BaselinePipeline` | Reference implementation; minimal code path for debugging |
| `DeterministicGraphPipeline` | Reproducible CI evals; deterministic fault classification and repair |
| `AgentPipeline` | Maximum flexibility; LLM decides which tools to call and when |

All three implement the same `Pipeline` protocol (`async def run(request) -> response`), so the API layer is identical regardless of which variant is active. The variant is chosen at startup via environment config.

---

## 5. Why LangGraph for the deterministic pipeline?

- **State management**: `TypedDict` state eliminates ad-hoc dicts; every field is typed
- **Conditional edges**: replace deeply nested `if/else` chains with declarative routing functions
- **Native LangSmith tracing**: graph steps appear as named spans with no extra instrumentation
- **Recursion limits**: prevent infinite repair loops without try/except scaffolding

The alternative — a hand-rolled `while retry < max_retries` loop — becomes hard to follow once there are two repair strategies (schema broadening vs. SQL logic repair).

---

## 6. Why TypedDict state over OOP state objects?

LangGraph node functions are stateless: `(state, services) -> state`. This design:

- Makes nodes independently unit-testable (just pass a dict)
- Avoids hidden instance state that complicates retry loops
- Allows LangGraph's checkpointing mechanism to serialize/deserialize state natively
- Enables parallel node execution without shared-mutable-state concerns

---

## 7. Why draft SQL + refine schema context?

The first LLM pass (draft SQL) identifies which tables are *actually needed* given the question. The `refine_schema_context` step then filters the schema context to only those tables before the final generation pass. This reduces:

- Token noise (irrelevant schema dilutes attention)
- Hallucination risk (fewer table choices → less ambiguity)

The second LLM pass receives a smaller, more focused prompt and consistently produces higher-quality SQL.

---

## 8. Why fault classification in the repair loop?

Treating all SQL failures the same wastes LLM calls:

- **`retrieval_fault`** (wrong/missing schema): fix by broadening retrieval (`top_k + 3`), then regenerate
- **`generation_fault`** (correct schema, wrong logic): fix by injecting a critique into the prompt

A classifier LLM call is cheaper than regenerating with the wrong strategy. Without classification, the repair loop always uses the same prompt repair path, which is ineffective when the root cause is missing schema context.

---

## 9. Why `asyncio.to_thread()` for ChromaDB and SQLite?

Neither ChromaDB's Python client nor SQLAlchemy's synchronous engine has a native async API. Options:

| Approach | Issue |
|---|---|
| `ThreadPoolExecutor` | Ad-hoc, not under asyncio's lifecycle, cancellation is awkward |
| `asyncio.to_thread()` | Uses the event loop's default executor, cancellable via `asyncio.wait_for`, no teardown needed |
| Fully async DB driver | Not available for ChromaDB; SQLAlchemy async requires a different engine setup |

`asyncio.to_thread()` keeps blocking I/O off the event loop without introducing executor management code. Wrapped with `asyncio.wait_for()`, it also enforces the per-query timeout.

---

## 10. Why dependency injection throughout?

All components (retrievers, generator, executor, LLM) are passed to constructors rather than instantiated inside methods. Benefits:

- **Tests**: swap in mocks without monkeypatching module globals
- **Implementations**: swap a retriever implementation without subclassing
- **Cross-cutting concerns**: wrap any component with a logging or metrics decorator transparently
- **Factory function**: a single `build_pipeline()` wires everything from config; callers don't need to know the internals

---

## 11. Why sqlglot for SQL validation?

- **Dialect-agnostic**: parses SQLite, PostgreSQL, MySQL, etc. with a single API
- **No database connection**: validation is pure parsing, usable in CI without a live DB
- **CTE-aware table extraction**: the schema recall metric correctly excludes CTE aliases from the set of "real" tables

The alternative — executing against SQLite to check validity — is slower and requires a database file for every schema.

---

## 12. Why execution accuracy over exact match as the primary metric?

Two SQL queries can be semantically identical but textually different:

```sql
-- Both return the same result set:
SELECT COUNT(*) FROM singer
SELECT COUNT(singer_id) FROM singer
```

Exact match would penalise the second query. Execution accuracy measures what the user actually cares about: do the result sets match? It is the primary metric used in academic Spider benchmarks and is more robust to equivalent reformulations.

Exact match is still reported as a secondary metric for debugging prompt formatting regressions.

---

## 13. Guardrails rationale

| Layer | Guard | Why |
|---|---|---|
| Input | Reject DDL/DML keywords | Prevent prompt injection via `DROP TABLE`, `INSERT`, etc. |
| SQL output | Block `sqlite_master` queries | Prevent schema leakage through system table access |
| Retrieval | Cap schema/example doc counts | Keep prompts within context window; avoid token waste |
| Output | Flag empty result sets | Surface uncertain answers rather than silently returning `[]` |

Guardrails are applied at well-defined pipeline checkpoints rather than scattered throughout node logic, making them easy to audit and test independently.
