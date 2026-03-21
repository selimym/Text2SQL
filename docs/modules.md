# Module Reference

Each module lives under `app/` and has a single clear responsibility.

---

## `app/core/` — Infrastructure

### `config.py`

`AppSettings` is the single source of truth for all configuration. It uses `pydantic-settings` to read values from environment variables (or a `.env` file). All settings have safe defaults so the app starts without a `.env`.

Key settings:

| Setting | Default | Purpose |
|---|---|---|
| `llm_provider` | `"anthropic"` | Which LLM to use (`"anthropic"` or `"openai"`) |
| `llm_model` | `"claude-3-5-sonnet-20241022"` | Model identifier passed to the provider |
| `embedding_provider` | `"openai"` | Which embeddings to use (`"openai"` or `"sentence-transformers"`) |
| `embedding_model` | `"text-embedding-3-small"` | Embedding model identifier |
| `chroma_persist_dir` | `".chroma"` | Where ChromaDB stores its data on disk |
| `spider_data_dir` | `"spider_data"` | Root directory of the Spider dataset |
| `max_result_rows` | `100` | Hard cap on rows returned by the executor |
| `query_timeout_seconds` | `30` | Timeout for SQL execution |

A singleton is available via `get_settings()`. See [configuration.md](configuration.md) for the full env var reference.

### `llm.py`

`get_llm(provider, model) -> BaseChatModel` is a factory that returns a LangChain `BaseChatModel`. Returning a common interface means `SQLGenerator` doesn't care whether the backend is Claude or GPT-4o.

### `logging.py`

`configure_logging(log_level)` sets up [structlog](https://www.structlog.org/) with ISO timestamps and console rendering. Called once at startup in `main.py`.

---

## `app/db/` — Database Layer

### `schema_doc.py`

Defines the Pydantic models that represent a table schema:

- **`ColumnInfo`** — name, data type, whether it's a PK, whether it's nullable
- **`ForeignKeyInfo`** — from_column → to_table.to_column
- **`SchemaDocument`** — one table: db_id, table_name, columns list, foreign keys list

`SchemaDocument.to_text()` serialises the schema into a human-readable string used for embedding and for inclusion in the LLM prompt.

### `spider_loader.py`

`load_schema_documents_from_spider(tables_json_path)` parses Spider's `tables.json` (which describes all databases in the dataset) and returns a flat list of `SchemaDocument` objects — one per table across all databases.

This is used by `scripts/index_spider.py` to build the schema vector store. See [spider-dataset.md](spider-dataset.md) for the full picture.

### `introspection.py`

`extract_schema(db_path)` dynamically introspects a SQLite database file using SQLAlchemy and returns `SchemaDocument` objects. This is an alternative to the Spider loader for databases not described in `tables.json`.

### `validator.py`

`SQLValidator.validate(sql) -> ValidationResult` runs the generated SQL through a series of checks using `sqlglot`:

1. Not empty
2. Syntactically valid SQL
3. Exactly one statement
4. A SELECT statement (no INSERT / UPDATE / DELETE)

Validation is intentionally strict and cheap — it runs before any database connection is opened.

### `executor.py`

`SQLExecutor.execute(sql, db_path) -> ExecutionResult` opens a SQLite connection, runs the query with:
- **Timeout protection** via `concurrent.futures.ThreadPoolExecutor`
- **Row cap** (returns at most `max_rows` rows)
- **Latency measurement** with `time.monotonic()`

`ExecutionResult` includes `success`, `rows`, `column_names`, `row_count`, `latency_ms`, and `error` / `error_category` on failure.

---

## `app/retrieval/` — Vector Search

### `example_doc.py`

`ExampleDocument` is the Pydantic model for a Spider training example stored in ChromaDB: `db_id`, `question`, `sql`, `tables_used`, `query_type`, `difficulty`.

`to_text()` returns `"Question: ...\nSQL: ..."` — the string that gets embedded.

### `embeddings.py`

`get_embeddings(provider, model) -> Embeddings` factory. Returns either `OpenAIEmbeddings` or `HuggingFaceEmbeddings` (sentence-transformers), both behind the LangChain `Embeddings` interface.

### `schema_retriever.py`

`SchemaRetriever` wraps a ChromaDB collection named `spider_schemas`.

- **`index(docs)`** — embeds each `SchemaDocument` via `to_text()` and upserts into ChromaDB. Document IDs are `{db_id}_{table_name}`.
- **`retrieve(question, db_id, top_k)`** — embeds the question and runs a similarity query filtered to `db_id`, returning the top-k most relevant `SchemaDocument` objects.

### `example_retriever.py`

`ExampleRetriever` wraps a ChromaDB collection named `spider_examples`.

- **`index(docs)`** — embeds each `ExampleDocument` via `to_text()` and upserts.
- **`retrieve(question, db_id, top_k)`** — same pattern as schema retriever, optionally filtered by `db_id`.

---

## `app/llm/` — Prompting and Generation

### `prompts.py`

`SYSTEM_PROMPT` tells the LLM it is an expert SQL generator and should return only valid SQL.

`build_user_prompt(question, schema_docs, example_docs)` assembles the full user-facing part of the prompt in three sections:

1. `## Schema Context` — one block per `SchemaDocument`
2. `## Similar Examples` — one block per `ExampleDocument` (omitted if none)
3. `## Question` — the raw question, followed by `SQL:` to prompt completion

### `generator.py`

`SQLGenerator.generate(user_prompt) -> str` calls `llm.invoke([system, user])` and strips any markdown code fences from the response before returning the SQL string.

---

## `app/pipeline/` — Orchestration

### `assembler.py`

`PromptAssembler.assemble(...)` is a thin wrapper around `build_user_prompt`. It exists to keep prompt-building logic separate from pipeline logic and to make it easy to swap in a different assembler later.

### `baseline.py`

`BaselinePipeline` is the core orchestrator. It wires all components together and implements the linear retrieval → generation → validation → execution flow.

Constructor parameters (all injected):
`schema_retriever`, `example_retriever`, `assembler`, `generator`, `validator`, `executor`, `spider_data_dir`

`run(request) -> QueryResponse` executes the full pipeline. If validation fails it returns early with a `validation_failed` flag in the response rather than raising an exception.

Database files are resolved as `{spider_data_dir}/{db_id}/{db_id}.sqlite`.

---

## `app/eval/` — Evaluation

See [spider-dataset.md](spider-dataset.md) for the full description of how evaluation works.

### `loader.py`

`load_dev_subset(spider_data_dir, db_filter, limit) -> list[EvalExample]` reads `dev.json` from the Spider dataset and returns a filtered, optionally capped list of `EvalExample` objects (each with `db_id`, `question`, `gold_sql`).

### `runner.py`

`run_evaluation(pipeline, ...)` iterates over dev examples, runs `pipeline.run()` for each, and computes an `EvalReport` with `total`, `execution_success`, `success_rate`, and `avg_latency_ms`.

The module also has a `__main__` entry point so it can be run directly:

```bash
uv run python -m app.eval.runner --split dev --limit 50
```

---

## `app/api/` — HTTP Interface

### `models.py`

| Model | Direction | Key fields |
|---|---|---|
| `QueryRequest` | Request | `question`, `db_id`, `top_k_schema` (5), `top_k_examples` (3) |
| `QueryResponse` | Response | `generated_sql`, `answer`, `execution_metadata`, `flags`, `trace_id` |
| `ExecutionMetadata` | Nested | `success`, `row_count`, `latency_ms`, `error` |
| `ErrorResponse` | Error | `error`, `detail`, `trace_id` |

### `routes.py`

Two endpoints:

- `GET /health` — returns `{"status": "ok"}`
- `POST /query` — accepts `QueryRequest`, delegates to `BaselinePipeline.run()`, returns `QueryResponse`

The `_pipeline` module-level variable is set via `set_pipeline()` during app lifespan in `main.py`.

---

## `app/main.py` — Application Entry Point

`create_app()` builds and returns the FastAPI application. The `@asynccontextmanager lifespan` handles startup:

1. Load `AppSettings`
2. Configure structlog
3. Initialise ChromaDB persistent client
4. Create embedding model
5. Create `SchemaRetriever` and `ExampleRetriever`
6. Create LLM
7. Assemble all pipeline components
8. Call `set_pipeline()` to register the pipeline with the router

Teardown is implicit — ChromaDB and LLM clients release resources when the process exits.

---

## `scripts/index_spider.py` — Indexing Script

One-time setup script that populates ChromaDB from the Spider dataset. Must be run before the server can answer queries.

Steps:
1. Load all `SchemaDocument` objects from `tables.json` → index into `spider_schemas`
2. Load training examples from `train_spider.json` → parse each SQL with `sqlglot` to extract table names → create `ExampleDocument` objects → index into `spider_examples`

```bash
uv run python scripts/index_spider.py
```
