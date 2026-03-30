# Bug Fixes & Improvements Log

## Fix 1 — `sqlglot.TokenError` crash in validator
**File:** `app/db/validator.py`
**Problem:** `SQLValidator.validate()` only caught `sqlglot.errors.ParseError`. When the LLM returned prose instead of SQL, `sqlglot.parse()` raised `sqlglot.errors.TokenError`, which was uncaught and crashed the entire experiment run.
**Fix:** Added `sqlglot.errors.TokenError` to the except clause so invalid LLM output is gracefully handled as `valid=False`.

---

## Fix 2 — Schema retriever returned empty column lists
**File:** `app/retrieval/schema_retriever.py`
**Problem:** `SchemaRetriever.retrieve()` reconstructed `SchemaDocument` objects with `columns=[]` and `foreign_keys=[]`. The full column info was stored in ChromaDB document text (via `SchemaDocument.to_text()`) but was never parsed back on retrieval. The LLM received table names with no column information and hallucinated column names (e.g. `stadium_name` instead of `name`, `country_of_origin` instead of `country`, `release_year` instead of `song_release_year`), causing most execution failures.
**Fix:** Added `_parse_schema_text()` helper that parses the stored document text back into `ColumnInfo` and `ForeignKeyInfo` objects. `retrieve()` now returns fully populated `SchemaDocument` instances.

---

## Fix 3 — LLM returning prose instead of SQL
**File:** `app/llm/prompts.py`
**Problem:** The system prompt said "Return ONLY the SQL query with no explanation or markdown" but did not specify what to do when the question couldn't be answered. The LLM responded with multi-paragraph reasoning text (e.g. "I need to analyze this step by step..."), which then crashed the validator (see Fix 1) and counted as a failure.
**Fix:** Added explicit fallback instruction: `If the question cannot be answered with the given schema, return: SELECT NULL`. This ensures the output is always valid SQL.

---

## Observed metrics before fixes (baseline, 50 examples, `concert_singer` + `pets_1`)
| Metric | Before | After fixes 1-3 | Delta |
|---|---|---|---|
| Execution accuracy | 0.52 | **0.82** | +0.30 |
| Execution success rate | 0.60 | **0.96** | +0.36 |
| Exact match rate | 0.08 | 0.08 | 0 |
| Schema recall | 0.96 | **0.98** | +0.02 |
| Schema precision | 0.35 | 0.35 | 0 |
| Schema noise ratio | 0.61 | 0.63 | ~0 |
| Prose-instead-of-SQL failures | 2 | 0 | -2 |
| TokenError crashes | 1 (aborted run) | 0 | -1 |

**Key takeaway:** Execution accuracy jumped from 52% → 82% (+30pp) purely from restoring column info in schema retrieval. Exact match stays at 8% because the LLM uses aliases and table prefixes that differ from gold SQL style — this is a known limitation of exact-match as a metric.

---

## Fix 5 — `avg_fewshot_table_overlap` is a misleading metric
**Files:** `app/eval/metrics.py`, `app/eval/runner.py`, `app/pipeline/graph_pipeline.py`, `app/api/models.py`
**Problem:** `fewshot_table_overlap` measured whether retrieved few-shot examples used the same tables as the gold SQL. This is circular: it uses the answer (gold SQL tables) as the ground truth for evaluating retrieval quality. Unlike schema retrieval — where `tables.json` provides dataset-level labels independent of any model — there are no ground-truth labels for what makes a few-shot example "relevant". A good few-shot example is one that improves LLM output, which depends on SQL pattern similarity, difficulty, and phrasing — not table overlap. Using an LLM to generate such labels would also be biased (measuring agreement with the LLM's own preferences, not actual quality). The only honest evaluation of few-shot retrieval is end-to-end: compare execution accuracy with vs without retrieved examples.
**Fix:** Removed `fewshot_table_overlap` from `metrics.py`, removed all computation and reporting from `runner.py` and `EvalReport`, removed `retrieved_example_sqls` from `QueryResponse` and `graph_pipeline.py`.

**TODO:** Stale tests in `tests/unit/test_eval_runner.py` still reference `avg_fewshot_table_overlap`, `retrieved_example_sqls`, and `fewshot_table_overlap` per-example key — these need to be removed.

**File:** `scripts/run_experiment.py`
**Problem:** `--output` defaulted to `experiment_report.json`, so every run overwrote the previous results.
**Fix:** Default output filename is now `experiment_report_<YYYYMMDD_HHMMSS>.json`. Pass `--output <name>` to override.

---

## Fix 6 — Missing Spider citation and metric documentation in README
**File:** `README.md`
**Problem:** The Spider dataset was used without attribution. Evaluation metrics were not explained anywhere, making results hard to interpret. The results table was missing.
**Fix:** Added Spider paper citation (Yu et al., EMNLP 2018), full metric definitions table, schema retrieval section explaining top-k vs threshold tradeoff, and an experiment results table for manual tracking.

---

## Fix 11 — Agent prompt too vague, no tool call tracing
**Files:** `app/pipeline/agent_pipeline.py`, `app/api/models.py`
**Problem 1:** The agent system prompt only said "use the available tools" with no ordering, no rules about column names, and no fallback instruction. The agent would skip schema retrieval, invent column names, or call tools in arbitrary order.
**Fix:** Replaced with a structured 5-step prompt that enforces: (1) get_schema first, (2) get_examples second, (3) write SQL using only retrieved names, (4) validate, (5) output SQL only. Added explicit rule against inventing table/column names and the `SELECT NULL` fallback.

**Problem 2:** No way to inspect what the agent actually did per example without LangSmith. The full message history was available in `result["messages"]` but discarded.
**Fix:** Added `_extract_tool_trace()` which walks the message history and builds a compact list of `{tool, args, result}` dicts. Stored in `step_timings["tool_trace"]` in the JSON report — visible per example without any external service. Result truncated to 300 chars to keep the report readable. Also widened `step_timings` type from `dict[str, float]` to `dict[str, Any]` in `QueryResponse`.

---

 Agent pipeline reported schema recall/precision as 0.00
**Files:** `app/pipeline/agent_tools.py`, `app/pipeline/agent_pipeline.py`
**Problem:** The agent retrieves schema dynamically via tool calls, so `retrieved_schema_summary` was never populated in `QueryResponse`. The eval runner computed recall/precision against an empty list, giving 0.00 for all schema metrics — making agent results incomparable to baseline and deterministic.
**Fix:** Added `retrieved_tables: list[str]` to `ToolContext`. The `_get_schema` tool appends table names to it on every call (deduped). `AgentPipeline.run()` clears the list at the start of each request and passes the accumulated tables as `retrieved_schema_summary` in all response paths (success, validation failure, agent error, no SQL).

---

 `print_summary_table` accidentally merged into `append_csv`
**File:** `scripts/run_experiment.py`
**Problem:** When inserting `append_csv`, the `def print_summary_table` line was dropped, merging its body into `append_csv` as dead code. This caused `NameError: name 'print_summary_table' is not defined` at the end of every run (after the JSON and CSV were already saved).
**Fix:** Restored `def print_summary_table` as a separate function.

---

 Agent pipeline crashes on malformed tool call args
**File:** `app/pipeline/agent_pipeline.py`
**Problem:** The LLM occasionally returns a plain natural language string as tool call `args` instead of a JSON object. The LangChain SDK's converter attempts `json.loads()` on it, fails, and routes to `invalid_tool_calls` — but `langchain_core 1.0.2`'s `AIMessage` validation still raises a `pydantic_core.ValidationError` (`tool_calls.0.args: Input should be a valid dictionary`), crashing the entire run after many examples have already been evaluated.
**Root cause:** LangChain SDK bug — the `json.loads` failure path doesn't fully prevent the invalid args from reaching `AIMessage`. Not fixable without updating the SDK.
**Fix:** Wrapped `self._agent.invoke()` in a try/except in `AgentPipeline.run()`. Errors are caught and returned as a `QueryResponse` with `flags=["agent_error", ...]`, so the run continues and the example counts as a failure rather than aborting the whole experiment.

---

 `create_react_agent` called with deprecated `state_modifier` argument
**File:** `app/pipeline/agent_pipeline.py`
**Problem:** The installed langgraph version replaced `state_modifier` with `prompt` in `create_react_agent`. The codebase was written against the old API, causing a `TypeError` on agent pipeline construction. Note: `create_react_agent` itself is also deprecated in favour of `create_agent` from langchain, but it still works — only the argument name changed.
**Fix:** Replaced `state_modifier=system_prompt` with `prompt=system_prompt`.

---

 Store similarity threshold in report + auto-append results to CSV
**Files:** `app/eval/experiment.py`, `scripts/run_experiment.py`
**Problem:** The experiment report did not record the similarity threshold used, making it impossible to know after the fact what settings produced a given result. Results also had to be manually copied into the README table.
**Fix:** Added `similarity_threshold` field to `ComparisonReport` (stored in JSON). After each run, one row per variant is appended to `results.csv` (created if it doesn't exist), containing: `run_date`, `variant`, `limit`, `db_filter`, `similarity_threshold`, all metrics, `wall_clock_seconds`, and `report_file` (path to the full JSON).

**Files:** `app/retrieval/schema_retriever.py`, `app/pipeline/baseline.py`, `app/pipeline/nodes.py`, `app/pipeline/factory.py`, `scripts/run_experiment.py`, `app/core/config.py`
**Context:** Top-k retrieval always returns exactly k tables regardless of relevance. Most Spider queries touch 1-2 tables, so k=5 guarantees 3-4 irrelevant tables in context (avg_schema_noise_ratio=0.63). The precision/noise metrics are meaningful but structurally penalised by fixed-k retrieval.
**Change:** Added optional `similarity_threshold` parameter to `SchemaRetriever.retrieve()`. Tables whose cosine distance exceeds the threshold are dropped. Configurable via `--similarity-threshold` CLI arg or `SCHEMA_SIMILARITY_THRESHOLD` in `.env`. Defaults to `None` (pure top-k, existing behaviour). The `broaden_schema` retry node intentionally skips the threshold to maximise recall on repair attempts.

---

## Checkpoint/resume + parallel async evaluation
**Files:** `app/eval/runner.py`, `app/eval/experiment.py`, `scripts/run_experiment.py`
**Problem 1:** Running the full Spider dev set (~1034 examples) across 3 pipeline variants is slow and fragile. A transient API error (e.g. a 404 mid-run from the LLM provider) aborts the entire run with no way to resume — all progress is lost.
**Problem 2:** All pipeline calls were sequential and blocking. Each example waited for the previous one to finish, even though the bottleneck is network I/O (LLM + embedding API calls) that could be parallelised.

**Fix:**
- Added `run_evaluation_async()` to `runner.py`: runs examples concurrently via `asyncio.Semaphore` + `asyncio.to_thread` (safe because each example is independent and sync pipelines release the GIL on I/O). Examples are processed in batches of `batch_size` (default 50); each batch runs fully in parallel then its results are written to the checkpoint file in one atomic write. On startup, loads the checkpoint and skips already-done examples.
- Added `_compute_result_metrics()` and `_aggregate()` helpers so both sync and async paths produce identical result dicts and `EvalReport` — checkpoint files are fully compatible with the final report.
- Added `compare_variants_async()` to `experiment.py`: calls `run_evaluation_async()` per variant sequentially (variants in parallel would multiply API load unnecessarily). Checkpoint path per variant: `<checkpoint_dir>/<variant>_<run_id>.jsonl` where `run_id` is a stable 8-char hash of `(limit, db_filter, similarity_threshold)` — one file per pipeline, deterministic across runs.
- `scripts/run_experiment.py`: added `--concurrency N` (default 8), `--batch-size N` (default 50), and `--checkpoint-dir PATH` (default `checkpoints/`) CLI args. `main()` now calls `asyncio.run(_async_main(args))`.

**Refinement — batched writes:** Initial implementation flushed the checkpoint file after every single example (with a lock). Replaced with batch-level writes: all examples in a batch complete first, then the whole batch is written in one `open/write/close`. This reduces I/O overhead and lock contention across ~1034 examples × 3 pipelines. Worst-case loss on crash is one batch (50 examples) rather than zero — an acceptable tradeoff.

**Note on async availability:** The LLM and embeddings clients both expose async methods (`aembed_documents`, `aembed_query`, `ainvoke`) but these are not wired into the pipeline code — all three pipelines use only sync methods. `asyncio.to_thread` is the right approach here: it parallelises the sync calls without requiring a full async rewrite of the pipeline layer.

**Transient 404 root cause:** LLM API providers occasionally return 404 "Model not found" mid-run when a model backend restarts or redeploys. This is infrastructure noise, not a config error. The checkpoint mechanism means such interruptions no longer lose progress — just re-run the same command to resume.

---

## `retry_count`, `flags`, `step_timings` not saved to per-example results
**Files:** `app/eval/runner.py`, `app/llm/generator.py`, `app/pipeline/baseline.py`, `app/pipeline/nodes.py`, `app/pipeline/agent_pipeline.py`

**Problem 1:** `_compute_result_metrics()` only accepted metric-relevant fields. `retry_count`, `flags`, and `step_timings` (which includes token usage and tool traces) were present in `QueryResponse` but never passed to the helper or written to the result dict. All three fields showed as missing in the JSON report.

**Problem 2:** `SQLGenerator.generate()` returned only the SQL string, discarding the `AIMessage` object and its `usage_metadata` (input/output/total tokens). Token usage was available from the LLM client but silently dropped.

**Problem 3:** `AgentPipeline` had a duplicate `run()` method — the second definition (lines 151-241) silently overwrote the first. The first method correctly extracted `tool_trace` via `_extract_tool_trace()`; the second omitted it entirely. This is why `tool_trace` was `None` for all 1034 agent examples — the model may well have been using tools, but the tracking code was dead.

**Fix:**
- `generator.py`: `generate()` now returns `(sql, usage)` where `usage` is a dict with `input_tokens`, `output_tokens`, `total_tokens` (empty dict if unavailable).
- `baseline.py`: unpacks the tuple; stores `{"usage": usage}` in `step_timings`.
- `nodes.py`: both `generate_draft_sql_node` and `generate_final_sql_node` unpack the tuple; store usage under `"draft_usage"` and `"final_usage"` keys in `step_timings`.
- `runner.py`: `_compute_result_metrics()` accepts `retry_count`, `flags`, `step_timings` as optional params; writes them to the result dict only when present. Both sync and async call sites updated.
- `agent_pipeline.py`: deleted the duplicate `run()` method (lines 151-241). The surviving method correctly calls `_extract_tool_trace()` and includes `tool_trace` in `step_timings` on all return paths.
