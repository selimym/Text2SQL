# The Spider Dataset

## What Is Spider?

[Spider](https://yale-lily.github.io/spider) is a large-scale, cross-domain Text-to-SQL benchmark created at Yale. It is the primary dataset used throughout this project for both building the retrieval index and running offline evaluation.

Key facts:

| Property | Value |
|---|---|
| Training examples | ~7,000 question→SQL pairs |
| Development examples | ~1,034 question→SQL pairs |
| Databases | 166 SQLite databases |
| Domains | Restaurants, sports, finance, education, government, … |
| SQL complexity | Easy, medium, hard, extra hard |

Each example contains:
- `db_id` — which database the question is about
- `question` — the natural-language question
- `query` — the gold-standard SQL answer

The dataset lives in `spider_data/` (gitignored — you need to download it separately).

Relevant files:

```
spider_data/
├── tables.json           # Schema definitions for all 166 databases
├── train_spider.json     # ~7k training examples
├── dev.json              # ~1k development/evaluation examples
└── database/
    ├── concert_singer/
    │   └── concert_singer.sqlite
    ├── pets_1/
    │   └── pets_1.sqlite
    └── ...               # one directory per db_id
```

---

## How We Use the Training Set

The training set (`train_spider.json`) is used in two ways: to build the **schema index** and the **few-shot example index**. Both are populated once by running the indexing script before the server starts.

### 1. Schema Index (from `tables.json`)

`tables.json` describes the schema of every database: tables, columns, data types, primary keys, and foreign keys. We load this with `spider_loader.py` to create `SchemaDocument` objects (one per table), then embed and store them in ChromaDB under the `spider_schemas` collection.

At query time, the question is embedded and ChromaDB returns the most relevant tables for the target `db_id`. This means the LLM prompt always contains accurate, up-to-date schema context — even for databases it has never seen during LLM pre-training.

### 2. Few-Shot Example Index (from `train_spider.json`)

Each training example is an `ExampleDocument` containing the question, the gold SQL, the `db_id`, and the tables referenced in the query. These are embedded using the text `"Question: ...\nSQL: ..."` and stored in ChromaDB under the `spider_examples` collection.

At query time, the question is embedded and the most similar examples (optionally filtered to the same `db_id`) are retrieved. These are inserted into the prompt as **few-shot demonstrations** — showing the LLM the style and structure of SQL expected for this type of question.

This is the "few-shot RAG" part of the system.

```
┌─────────────────────────────────────────────┐
│  Indexing (run once via index_spider.py)    │
│                                             │
│  tables.json ──► SchemaRetriever.index()    │
│                       └─► spider_schemas    │  (ChromaDB)
│                                             │
│  train_spider.json ──► ExampleRetriever.index()
│                            └─► spider_examples  │  (ChromaDB)
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Inference (every /query request)           │
│                                             │
│  question ──► embed ──► spider_schemas      │
│                    └─► top-k SchemaDocuments│
│                                             │
│  question ──► embed ──► spider_examples     │
│                    └─► top-k ExampleDocuments│
└─────────────────────────────────────────────┘
```

### Why embed `"Question: ...\nSQL: ..."` for examples?

When a new question arrives at inference time, we embed just the question and search for it. For this similarity search to work well, the stored embeddings must capture what the question is *about*, not just the SQL syntax. Embedding the question-SQL pair together means the vector encodes both the intent and the query pattern — similar questions will cluster together even if they use different SQL constructs.

---

## Offline Evaluation

We evaluate against the Spider **dev set** (`dev.json`), which has ~1,034 examples across all databases.

### What We Currently Measure

The evaluation harness (`app/eval/runner.py`) runs the full pipeline — retrieval, generation, validation, execution — on a subset of dev examples and reports:

| Metric | Definition |
|---|---|
| `execution_success` | Number of queries that executed without error |
| `success_rate` | `execution_success / total` |
| `avg_latency_ms` | Mean SQL execution time |

**Important:** this measures *execution success*, not *answer correctness*. A query counts as a success if it ran without error, regardless of whether the returned rows match the gold answer. True accuracy (comparing result sets to gold) is a planned metric for a later milestone.

### Running the Evaluation

```bash
# Run on 50 examples from the full dev set
uv run python -m app.eval.runner --split dev --limit 50

# Run on specific databases only
uv run python -m app.eval.runner --split dev --limit 100 --db-filter concert_singer pets_1

# Save results to a file
uv run python -m app.eval.runner --split dev --limit 50 --output results.json
```

### What Happens Internally

1. `EvalLoader.load_dev_subset()` reads `dev.json`, optionally filters by `db_id`, and caps the list at `limit`.
2. For each `EvalExample`, a `QueryRequest` is constructed (same format as an API request).
3. `BaselinePipeline.run()` is called — the full retrieval → generation → validation → execution flow runs.
4. Success is determined by `execution_metadata.success`.
5. Results are aggregated into an `EvalReport`.

The harness imports `BaselinePipeline` directly — no HTTP server needed. This makes it fast to iterate on pipeline changes and run evaluation in CI.

### Known Limitations of the Current Metric

- A query that returns wrong rows but doesn't error is counted as a success.
- A query that is semantically correct but times out is counted as a failure.
- No partial credit for queries that are structurally close to the gold SQL.

These will be addressed by adding result-set comparison in a future milestone.
