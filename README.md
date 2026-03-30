# Text2SQL

Text-to-SQL backend using Dual RAG (schema + few-shot retrieval), FastAPI, ChromaDB, and a provider-agnostic LLM layer.

## Prerequisites

- Python 3.12 (`python` must point to 3.12 — verify with `python --version`)
- `uv` installed
- An LLM provider API key (OpenAI, Anthropic, or any compatible provider)
- Spider dataset downloaded and placed under `spider_data/` (see [docs/spider-dataset.md](docs/spider-dataset.md))

## Quickstart

### 1. Install dependencies

```bash
uv sync
python -m pip install -e . --no-deps
```

> `uv sync` installs all dependencies from `pyproject.toml` into `.venv`. `pip install -e . --no-deps` registers the `app` package so Python can import it without re-resolving dependencies.

### 2. Activate the venv

```bash
source .venv/bin/activate
```

### 3. Create your `.env`

```bash
cp .env.example .env
```

Edit `.env` to set your LLM provider and API key. See [docs/configuration.md](docs/configuration.md) for all options.

### 4. Index the Spider dataset (run once)

```bash
python scripts/index_spider.py
```

This embeds all schemas and training examples into ChromaDB (stored in `.chroma/`). Only needs to be re-run if the dataset or embedding model changes.

### 5. Run an experiment

```bash
python scripts/run_experiment.py --variants baseline deterministic --limit 50
```

Results are printed as a summary table and appended to `results.csv`. Full JSON report is saved to `experiment_report_<timestamp>.json`.

---

## Common arguments for `run_experiment.py`

| Argument | Default | Description |
|---|---|---|
| `--variants` | `baseline deterministic agent` | Pipeline variants to compare |
| `--limit` | _(all)_ | Max number of Spider dev examples to evaluate |
| `--db-filter` | _(all)_ | Restrict to specific database IDs |
| `--output` | `experiment_report_<timestamp>.json` | Output file for full JSON report |
| `--similarity-threshold` | _(none)_ | Cosine distance cutoff for schema retrieval |
| `--concurrency` | `8` | Max parallel examples (async evaluation) |
| `--batch-size` | `50` | Examples per checkpoint write |
| `--checkpoint-dir` | `checkpoints/` | Directory for resume checkpoints |

**Recommended first run:** `--variants baseline deterministic --limit 50`
Skip `agent` until you've confirmed the basics work — it makes many more LLM calls per example.

Long runs (full Spider dev set, ~1034 examples) are resumable: if interrupted, re-run the same command and it picks up from the last checkpoint.

---

## Architecture

Three pipeline variants are available, all satisfying the same `Pipeline` protocol. See [docs/architecture.md](docs/architecture.md) for the full component diagram and data flow.

| Variant | Description | LLM calls |
|---|---|---|
| `baseline` | Linear 6-step pipeline: retrieve schema → retrieve examples → assemble prompt → generate SQL → validate → execute | 1 per query |
| `deterministic` | Same as baseline but wrapped in a LangGraph state machine with a critique/repair loop when SQL fails | 1–3 per query |
| `agent` | ReAct agent that autonomously decides which tools to call (get_schema, get_examples, validate_sql, execute_sql) | Dynamic |

The LLM and embedding backends are fully provider-agnostic — switching providers is a single config change. See [docs/configuration.md](docs/configuration.md).

---

## Evaluation metrics

| Metric | What it measures |
|---|---|
| **execution_accuracy** | Generated SQL runs AND returns the same result set as gold SQL (order-insensitive). Primary metric. |
| **exact_match_rate** | Token-for-token identical to gold after normalization. Very strict — semantically equivalent queries fail. Useful only as a lower bound. |
| **success_rate** | SQL is syntactically valid and executes without error, regardless of correctness. |
| **avg_schema_recall** | Fraction of tables needed by gold SQL that were retrieved. High = found the right tables. |
| **avg_schema_precision** | Fraction of retrieved tables that were actually needed. Low = noisy context sent to the LLM. |
| **avg_schema_noise_ratio** | `1 - precision`. Fraction of retrieved tables that are irrelevant. |

> **Note on exact match:** Exact match stays low (~0.08–0.20) even when execution accuracy is high because the LLM uses aliases, table prefixes, and formatting that differ from gold SQL style. This is expected — don't optimise for exact match at the expense of execution accuracy.

---

## Schema retrieval

Schema retrieval uses ChromaDB vector search filtered by `db_id`. By default it returns the top-k most similar tables (`top_k_schema=5`). This always retrieves exactly k tables regardless of relevance, which inflates noise on simple queries (avg noise ratio ~0.63 with k=5 on Spider).

A **similarity threshold** can be set to drop tables whose cosine distance exceeds the cutoff:

```bash
python scripts/run_experiment.py --variants baseline --limit 50 --similarity-threshold 0.5
# or via .env:
SCHEMA_SIMILARITY_THRESHOLD=0.5
```

The threshold is embedding-model-dependent and must be calibrated empirically. The deterministic pipeline's `refine_schema_context` node achieves near-perfect precision (0.98) without a threshold by using the draft SQL to identify which tables are actually needed — a two-pass approach that is more robust than distance-based filtering. See [LEARNINGS.md](LEARNINGS.md) §5–6 for the precision/recall tradeoff analysis.

---

## Experiment results

Results on Spider dev set (full ~1034 examples unless noted):

| Date | Variant | Limit | Threshold | Exec Acc | Exact Match | Schema Recall | Schema Precision | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-03-26 | baseline | 50 | none | 0.52 | 0.08 | 0.96 | 0.35 | Before fixes — columns=[] bug |
| 2026-03-26 | baseline | 50 | none | 0.82 | 0.08 | 0.98 | 0.35 | After fixes (validator, schema retriever, prompt) |
| 2026-03-26 | baseline | all | none | 0.75 | 0.11 | 0.98 | 0.42 | Full dev set |
| 2026-03-26 | deterministic | all | none | 0.76 | 0.17 | 0.99 | 0.94 | Refine step achieves near-perfect precision |
| 2026-03-26 | agent | all | none | 0.72 | 0.10 | 0.63 | 0.25 | Model makes few tool calls, answers from memory |

Key observations:
- The **30pp accuracy gap** (0.52 → 0.82) before/after fixes was caused entirely by a silent data pipeline bug (schema retriever returning empty column lists), not by the LLM. See [FIXES.md](FIXES.md) Fix 2.
- The **deterministic pipeline** achieves schema precision of 0.94 vs 0.42 for baseline — without any similarity threshold — because the `refine_schema_context` node uses the draft SQL to narrow context.
- The **agent pipeline** underperforms baseline (0.72 vs 0.75) because the model makes few tool calls and answers from parametric memory without schema context. Tool use must be verified before building an agent pipeline. See [LEARNINGS.md](LEARNINGS.md) §8.
- **Exact match** is not a useful optimisation target — it penalises stylistically different but semantically correct SQL.

Results are auto-appended to `results.csv` after each run.

---

## Dataset

This project uses the **Spider** benchmark dataset for training examples and evaluation.

> Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., & Radev, D. (2018).
> **Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.**
> *EMNLP 2018.* https://arxiv.org/abs/1809.08887

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — component diagram, data flow, pipeline variants
- [docs/configuration.md](docs/configuration.md) — all environment variables and provider options
- [docs/spider-dataset.md](docs/spider-dataset.md) — dataset setup and evaluation details
- [docs/modules.md](docs/modules.md) — module-level API reference
- [FIXES.md](FIXES.md) — full log of bugs found and fixed
- [LEARNINGS.md](LEARNINGS.md) — lessons learned building and debugging the system
- [CLAUDE.md](CLAUDE.md) — development workflows and conventions
