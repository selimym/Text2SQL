# Text2SQL

Text-to-SQL backend using Dual RAG (schema + few-shot retrieval), FastAPI, ChromaDB, and provider-agnostic LLM.

## Quickstart

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in your API key:**
   Edit `.env` and add either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

3. **Install dependencies:**
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Start the server:**
   ```bash
   uv run uvicorn app.main:app --reload
   ```

5. **Test the health endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```

## Architecture

The project is organized by module:
- **Core:** Configuration, logging, and LLM provider factory
- **Database:** Schema introspection, Spider dataset loader, validator, and query executor
- **Retrieval:** ChromaDB-backed schema and example retrievers
- **LLM:** Prompt templates and SQL generation logic
- **Pipeline:** Prompt assembly and deterministic baseline pipeline
- **Evaluation:** Spider dataset utilities and offline evaluation runner
- **API:** FastAPI routes and request/response models

See [CLAUDE.md](CLAUDE.md) for more details on development workflows and conventions.
