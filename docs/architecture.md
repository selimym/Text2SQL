# Architecture Overview

## What This System Does

Text2SQL takes a natural-language question and a database identifier (`db_id`), and returns a SQL SELECT query along with its execution results. The key challenge is that the system needs to know which tables and columns exist in the target database, and ideally have seen similar questions before — that's where **Dual RAG** comes in.

## Dual RAG in a Nutshell

RAG = Retrieval-Augmented Generation. Instead of sending a bare question to the LLM, we first retrieve relevant context from vector stores and include it in the prompt. This system uses **two** separate retrieval passes:

1. **Schema retrieval** — finds the table schemas most relevant to the question (so the LLM knows the structure of the database)
2. **Example retrieval** — finds similar question→SQL pairs from the Spider training set (so the LLM can follow patterns it has seen before)

Both retrievers are backed by ChromaDB, a local vector database.

## High-Level Component Diagram

```
HTTP Request (POST /query)
        │
        ▼
  BaselinePipeline
        │
        ├─── SchemaRetriever ──► ChromaDB (spider_schemas)
        │        (top-k tables for this db_id)
        │
        ├─── ExampleRetriever ──► ChromaDB (spider_examples)
        │        (top-k similar Q→SQL pairs)
        │
        ├─── PromptAssembler
        │        builds: schema context + examples + question
        │
        ├─── SQLGenerator ──► LLM (Claude / GPT-4o)
        │        returns raw SQL string
        │
        ├─── SQLValidator
        │        checks: non-empty, parseable, SELECT-only
        │
        └─── SQLExecutor ──► SQLite database file
                 returns: rows, column names, latency
```

## End-to-End Data Flow

Given `question = "How many singers are there?"` and `db_id = "concert_singer"`:

### 1. Schema retrieval

The question is embedded into a vector. ChromaDB returns the most semantically similar table schemas from the `concert_singer` database — e.g. the `singer` table schema. These arrive as `SchemaDocument` objects.

### 2. Example retrieval

Same embedding is used to query the examples collection, filtered to `concert_singer`. Returns similar question→SQL pairs from the Spider training set, e.g.:

```
Q: What are the names of all singers?
SQL: SELECT name FROM singer
```

### 3. Prompt assembly

The retrieved context is combined into a structured prompt:

```
## Schema Context
Table: singer
Columns: singer_id (int) [PK], name (text), country (text), ...

## Similar Examples
Q: What are the names of all singers?
SQL: SELECT name FROM singer

## Question
How many singers are there?

SQL:
```

### 4. SQL generation

The LLM receives the system prompt ("You are an expert SQL generator...") and the assembled user prompt. It returns something like:

```sql
SELECT COUNT(*) FROM singer
```

Markdown code fences are stripped if the model wraps the SQL.

### 5. Validation

`SQLValidator` runs the SQL through `sqlglot` to check:
- it is syntactically valid
- it contains exactly one statement
- it is a SELECT statement (no writes allowed)

If validation fails, the pipeline returns early with a `validation_failed` flag rather than crashing.

### 6. Execution

`SQLExecutor` connects to `spider_data/concert_singer/concert_singer.sqlite` and runs the query with a timeout (default: 30s) and row cap (default: 100 rows). Latency is recorded.

### 7. Response

```json
{
  "question": "How many singers are there?",
  "generated_sql": "SELECT COUNT(*) FROM singer",
  "answer": "[[76]]",
  "retrieved_schema_summary": ["singer", "song"],
  "execution_metadata": { "success": true, "row_count": 1, "latency_ms": 12.4 }
}
```

## Key Design Decisions

**Provider-agnostic LLM and embeddings.** Both the LLM (`get_llm()`) and embedding model (`get_embeddings()`) are created through factory functions. Switching from Claude to GPT-4o or from OpenAI embeddings to a local sentence-transformers model is a single config change.

**Dependency injection throughout.** `BaselinePipeline` receives all its components via constructor. This makes unit testing straightforward — tests can inject mocks for the LLM, ChromaDB, and executor without standing up real services.

**Pipeline Protocol.** All three variants satisfy the `Pipeline` Protocol defined in `app/pipeline/protocol.py`. This means the API server, eval runner, and experiment runner can work with any variant interchangeably. Select the variant at startup via `PIPELINE_VARIANT`.

**Validation before execution.** SQL is validated before it ever touches a database, preventing syntax errors from reaching the executor and ensuring the system only runs SELECT queries.

**Offline evaluation.** The eval harness runs the full pipeline against the Spider dev set without a live server — it imports `BaselinePipeline` directly. See [spider-dataset.md](spider-dataset.md) for how evaluation works.

## Pipeline Variants

As of Milestone B, the system supports three pipeline variants selectable via `PIPELINE_VARIANT`:

| Variant | Class | Flexibility | Observability | LLM Calls |
|---|---|---|---|---|
| `baseline` | `BaselinePipeline` | Fixed 6-step linear | None (no tracing) | 1 per query |
| `deterministic` | `DeterministicGraphPipeline` | Fixed topology + repair loop | LangSmith + step_timings | 1–3 per query |
| `agent` | `AgentPipeline` | LLM decides tool usage | LangSmith + retry_count | Dynamic |

**Variant A — Baseline** (`baseline`): A simple linear pipeline defined in `app/pipeline/baseline.py`. Retrieves schema → retrieves examples → assembles prompt → generates SQL → validates → executes. No retry mechanism. The reference implementation.

**Variant B — Deterministic Graph** (`deterministic`): A LangGraph `StateGraph` defined in `app/pipeline/graph_pipeline.py`. Follows the same 6-step sequence as the baseline but adds a critique/repair loop when SQL fails validation or execution.

### Variant B Graph Topology

```
START
  └─► retrieve_schema
       └─► retrieve_examples
            └─► assemble_prompt ◄──────────────────────────┐
                 └─► generate_sql                           │ (generation_fault)
                      └─► validate_sql                      │
                           ├─[valid]──► execute_sql         │
                           │             ├─[success]──► build_response ──► END
                           │             └─[failure]──► critique_failure ─┤
                           └─[invalid]──► critique_failure  │              │
                                          ├─[retrieval_fault]►broaden_schema┘
                                          └─[generation_fault]─────────────┘
                           (budget exhausted at any point) ──► build_response ──► END
```

`critique_failure` increments `retry_count` and classifies the failure as either a `retrieval_fault` (wrong/missing schema context) or `generation_fault` (correct schema but wrong SQL logic). The repair loop is bounded by `MAX_RETRIES` (default: 2).

**Variant C — ReAct Agent** (`agent`): A `create_react_agent` from LangGraph prebuilt, defined in `app/pipeline/agent_pipeline.py`. The LLM receives a human message with the question and database ID, then autonomously decides which tools to call and how many times.

### Agent Tools (Variant C)

| Tool | Purpose | Returns |
|---|---|---|
| `get_schema` | Retrieve relevant schema for a question | Formatted table/column text |
| `get_examples` | Retrieve similar SQL examples | Formatted Q→SQL pairs |
| `validate_sql` | Check SQL syntax | `"valid"` or `"invalid: <reason>"` |
| `execute_sql` | Run SQL against the database | JSON with `success`, `rows`, `column_names` |

The agent terminates when the LLM responds with only SQL (no tool calls). The final SQL is then validated and executed once by `AgentPipeline.run()` before constructing the `QueryResponse`.

### Running Experiments

Use `scripts/run_experiment.py` to compare all three variants on a Spider subset:

```bash
uv run python scripts/run_experiment.py \
  --variants baseline deterministic agent \
  --limit 20 --db-filter concert_singer \
  --output experiment_report.json
```

Enable LangSmith tracing by setting `LANGCHAIN_API_KEY` — the script automatically enables tracing when the key is present.
