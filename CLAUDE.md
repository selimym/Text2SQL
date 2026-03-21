# Text2SQL — Claude Code Context

## Project
Text-to-SQL backend using Dual RAG (schema + few-shot retrieval), FastAPI, ChromaDB, and provider-agnostic LLM.

## Commands
- Run tests: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Format: `uv run ruff format .`
- Type check: `uv run mypy app`
- Start server: `uv run uvicorn app.main:app --reload`
- Index Spider data: `uv run python scripts/index_spider.py`
- Run evaluation: `uv run python -m app.eval.runner --split dev --limit 50`

## Conventions
- Feature branches per module, PRs at module boundaries, small commits per logical unit
- Docstrings only for non-obvious logic
- All config via pydantic-settings (AppSettings), never hardcode secrets
- Tests mock LLM and ChromaDB; only executor tests use a real SQLite fixture
- Type annotations required on all public functions
- `uv run` prefix for all commands

## Architecture
- `app/core/`: config, logging, LLM factory
- `app/db/`: schema models, introspection, Spider loader, validator, executor
- `app/retrieval/`: ChromaDB-backed schema and example retrievers
- `app/llm/`: prompt templates, SQL generator
- `app/pipeline/`: prompt assembler, deterministic baseline pipeline
- `app/eval/`: Spider dataset loader, offline eval runner
- `app/api/`: FastAPI routes and Pydantic request/response models
