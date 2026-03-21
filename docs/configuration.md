# Configuration

All settings are managed by `AppSettings` in `app/core/config.py` using [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/). Values are read from environment variables or a `.env` file at project root. Copy `.env.example` to `.env` to get started.

## LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Which LLM backend to use. Options: `anthropic`, `openai` |
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | Model identifier passed to the provider |
| `ANTHROPIC_API_KEY` | _(empty)_ | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | _(empty)_ | Required when `LLM_PROVIDER=openai` |

## Embeddings

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | Which embedding model to use. Options: `openai`, `sentence-transformers` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model identifier |

When using `sentence-transformers`, `EMBEDDING_MODEL` should be a HuggingFace model name (e.g. `all-MiniLM-L6-v2`). No API key required — the model runs locally.

## Vector Store

| Variable | Default | Description |
|---|---|---|
| `CHROMA_PERSIST_DIR` | `.chroma` | Directory where ChromaDB stores its data. Relative to project root. |

The ChromaDB directory must exist and be writable. On first run after indexing, it will contain two collections: `spider_schemas` and `spider_examples`.

## Dataset

| Variable | Default | Description |
|---|---|---|
| `SPIDER_DATA_DIR` | `spider_data` | Root directory of the Spider dataset. See [spider-dataset.md](spider-dataset.md). |

## Query Execution

| Variable | Default | Description |
|---|---|---|
| `MAX_RESULT_ROWS` | `100` | Hard cap on rows returned per query. Prevents large result sets. |
| `QUERY_TIMEOUT_SECONDS` | `30` | Maximum time allowed for a single SQL query to execute. |

## Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Minimum log level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Pipeline Variants

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_VARIANT` | `baseline` | Which pipeline variant to use. Options: `baseline`, `deterministic`, `agent` |
| `MAX_RETRIES` | `2` | Maximum repair cycles for the deterministic variant |
| `AGENT_MAX_ITERATIONS` | `10` | Maximum tool-call rounds for the agent variant |
| `LANGSMITH_PROJECT` | `text2sql` | LangSmith project name for tracing |

### LangSmith Tracing

LangChain traces all LLM calls automatically when the following variables are set:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your LangSmith key>
LANGCHAIN_PROJECT=text2sql   # optional; defaults to LANGSMITH_PROJECT setting
```

The `run_experiment.py` script enables tracing automatically if `LANGCHAIN_API_KEY` is present in the environment.

## Minimal `.env` to get started

```dotenv
# Required: pick one provider and supply its key
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic

# Required if using OpenAI embeddings (default)
OPENAI_API_KEY=sk-...

# Or switch to local embeddings (no API key needed)
# EMBEDDING_PROVIDER=sentence-transformers
# EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Provider Combinations

| LLM | Embeddings | API keys needed |
|---|---|---|
| Anthropic | OpenAI | `ANTHROPIC_API_KEY` + `OPENAI_API_KEY` |
| OpenAI | OpenAI | `OPENAI_API_KEY` |
| Anthropic | sentence-transformers | `ANTHROPIC_API_KEY` only |
| OpenAI | sentence-transformers | `OPENAI_API_KEY` only |

sentence-transformers downloads the model on first use and caches it locally. Useful for development without an OpenAI key.
