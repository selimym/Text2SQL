# Agentic Workflow Learnings — Text2SQL Project

A running log of lessons learned building and debugging a Text-to-SQL system with RAG, LangGraph, and a tool-calling agent. Written to be useful beyond this specific project.

---

## 1. Always verify what your retriever actually returns

**What happened:** The schema retriever stored full column information in ChromaDB document text, but reconstructed `SchemaDocument` objects with `columns=[]` on retrieval. The LLM received table names with no column info and hallucinated plausible-sounding column names (`stadium_name`, `country_of_origin`, `release_year`).

**Impact:** Execution accuracy was 0.52 instead of 0.82 — a 30pp gap caused entirely by a silent data pipeline bug, not by the LLM.

**General lesson:** In RAG pipelines, the retriever is a data transformation step. Bugs there are silent — the LLM will not tell you it received incomplete context, it will just make things up. Always add a debug path to print exactly what the LLM receives before blaming the model.

**Pattern to adopt:** After building any retriever, write a one-off check that retrieves a known example and asserts the output contains what you expect. Don't assume the round-trip (index → store → retrieve → reconstruct) is lossless.

---

## 2. LLMs need explicit fallback instructions, not just positive instructions

**What happened:** The system prompt said "Return ONLY the SQL query". When the LLM couldn't answer a question, it returned multi-paragraph reasoning text instead of SQL. This crashed the validator downstream.

**General lesson:** Prompts that only describe the happy path leave the model to improvise on edge cases. Always specify what to do when the task cannot be completed. `"If you cannot answer, return SELECT NULL"` is one sentence that eliminates an entire failure mode.

**Pattern to adopt:** For every output format constraint in your prompt, ask: "what will the model do if it genuinely can't produce this output?" Add an explicit instruction for that case.

---

## 3. Catch exceptions at every pipeline boundary, not just at the top

**What happened:** `sqlglot.parse()` raised `TokenError` (not `ParseError`) when given prose text. The validator only caught `ParseError`, so the exception propagated up and aborted the entire 50-example evaluation run after partial completion.

**General lesson:** In evaluation loops, a single unhandled exception destroys all accumulated results. Each pipeline stage should catch its own failure modes and return a structured error result rather than raising. The outer loop should never crash — it should record the failure and continue.

**Pattern to adopt:** Wrap each pipeline stage in a try/except that returns a typed failure result. Reserve top-level exceptions for truly unrecoverable errors (missing config, bad credentials). For anything that can vary per input (LLM output, SQL parsing, tool calls), catch and record.

---

## 4. Metrics can be structurally misleading — understand what they actually measure

**What happened:** `avg_fewshot_table_overlap` measured whether retrieved few-shot examples used the same tables as the gold SQL. It always read 0.0 for the baseline pipeline (which didn't populate the field), and would have been circular even if it had: it uses the answer to evaluate the retrieval that was supposed to help find the answer.

**General lesson:** A metric that looks reasonable can be measuring the wrong thing. Before adding a metric, ask: (a) what ground truth does it compare against, (b) where does that ground truth come from, and (c) is that source independent of the system being evaluated? If the ground truth comes from the model itself or from the answer you're trying to predict, the metric is circular.

**For retrieval metrics specifically:** Schema retrieval can be evaluated honestly because `tables.json` provides correct table labels independently of any model. Few-shot retrieval cannot be evaluated this way — there are no ground-truth labels for "which example is most helpful". The only honest evaluation is end-to-end: run with vs without retrieved examples and compare execution accuracy.

---

## 5. Top-k retrieval is a blunt instrument — understand the precision/recall tradeoff

**What happened:** With `top_k=5` on databases with 4 tables, the retriever always returned more tables than needed. Schema noise ratio was 0.63 — 63% of retrieved context was irrelevant. Adding a similarity threshold of 0.8 collapsed recall to 0.63 (too aggressive). A threshold around 1.2–1.4 achieved near-perfect precision (0.98) with full recall (1.0).

**General lesson:** Fixed-k retrieval is a ceiling on precision. If your queries typically need 1-2 items but you always retrieve 5, you're guaranteed noise. A similarity threshold adapts to query complexity but is embedding-model-dependent and must be calibrated empirically.

**Key insight about cosine distance:** ChromaDB uses cosine distance (0=identical, 2=opposite). In practice with text embeddings, everything lands between 0.2 and 1.2. The gap between "relevant" and "irrelevant" items in your specific embedding space may be small — if it is, no threshold will cleanly separate them and you need better representations instead.

**Pattern to adopt:** Before tuning thresholds, extract the actual distance distribution for relevant vs irrelevant items on a sample. If the distributions overlap heavily, fix the representation (richer document text, better embedding model) rather than tuning the cutoff.

---

## 6. The deterministic pipeline's schema refinement step is doing real work

**What happened:** The deterministic pipeline (LangGraph with retry loop) achieved schema precision of 0.98 vs 0.35 for baseline — without any similarity threshold. The `refine_schema_context` node narrows the schema after a draft SQL is generated, using the draft to identify which tables are actually needed.

**General lesson:** A two-pass approach (broad retrieval → draft → refine) can achieve better precision than threshold filtering, because the draft SQL provides a strong signal about which tables are relevant. This is more robust than distance-based filtering because it uses semantic understanding rather than embedding geometry.

**Tradeoff:** It costs an extra LLM call per example. For this project the accuracy gain justified it. For latency-sensitive production use, the threshold approach is cheaper.

---

## 7. Execution accuracy and exact match measure different things — don't conflate them

**What happened:** Exact match stayed at 0.08 even when execution accuracy was 0.82. The LLM consistently wrote `COUNT(*) AS total_singers` instead of bare `count(*)`, used table aliases, added semicolons — all semantically identical but string-different from gold SQL.

**General lesson:** Exact match is a proxy metric that penalises stylistic differences. It's useful as a lower bound and for catching regressions, but a low exact match with high execution accuracy is not a problem — it means your model writes correct SQL in a different style than the benchmark authors. Don't optimise for exact match at the expense of execution accuracy.

**Exception:** If exact match is very low AND execution accuracy is also low, it may indicate the model is generating structurally wrong SQL rather than just stylistically different SQL. Use it as a diagnostic, not a target.

---

## 8. Model capability for tool use must be verified before building an agent pipeline

**What happened:** The `Turbo` model was used with a LangGraph ReAct agent. Across 50 examples, it made 0 tool calls — it answered entirely from parametric memory, ignoring the ReAct loop. In one case it output raw `<tool_call>` syntax as plain text instead of invoking the tool. The agent performed worse than the baseline (0.72 vs 0.82 execution accuracy) because it had no schema context.

**General lesson:** Not all LLMs support tool use in the format a given framework expects. LangGraph's ReAct agent uses a specific message format for tool invocation. A model that wasn't fine-tuned for this format will either ignore tools entirely or output tool call syntax as text. Before building an agent pipeline, verify with a minimal test (one tool, one question) that the model actually invokes tools.

**How to detect this quickly:** After one agent run, check `tool_trace` in the results. If it's empty across all examples, the model is not using tools. Don't wait for a full evaluation run to discover this.

---

## 9. Agent prompts need explicit step-by-step structure, not just capability descriptions

**What happened:** The original agent prompt said "use the available tools to retrieve schema information and examples, validate SQL, and execute queries." The model ignored it. The revised prompt gave numbered steps with explicit ordering and a rule against inventing column names.

**General lesson:** Vague capability descriptions ("you can use tools X, Y, Z") don't constrain agent behaviour. Agents need procedural prompts: numbered steps, explicit ordering, rules about what not to do, and a defined termination condition. The more the model has to infer about the intended workflow, the more it will deviate.

**Specific patterns that help:**
- Number the steps explicitly (1, 2, 3...)
- State what to do first, not just what's available
- Add negative constraints ("use ONLY names from the schema you retrieved, never invent names")
- Define the termination condition explicitly ("your final answer must be ONLY the SQL query")
- Add a fallback for the unanswerable case

---

## 10. Track what the agent actually did, not just what it produced

**What happened:** The first agent runs showed 0.76 execution accuracy but there was no way to know whether the agent had called tools, in what order, or what it received back — without LangSmith. The full message history was available in `result["messages"]` but was being discarded.

**General lesson:** Agent evaluation without execution traces is debugging blind. You can see the output was wrong but not why. The message history in LangGraph contains everything: which tools were called, with what arguments, and what they returned. Extracting a compact trace costs almost nothing and makes failures immediately diagnosable.

**Implementation:** Walk `result["messages"]`, collect `AIMessage` tool calls and `ToolMessage` responses into a list of `{tool, args, result}` dicts. Store in the per-example result. This gives you LangSmith-level visibility with zero external dependencies.

---

## 11. Dependency version mismatches between internal SDKs and open-source packages are a recurring risk

**General lesson:** Internal SDKs are built and tested against specific versions of their open-source dependencies. When you install the internal SDK alongside newer open-source packages, you get silent incompatibilities that only surface at runtime on specific code paths. The second bug only appeared when the model actually tried to make a tool call — it would never have been caught in unit tests.

**Patterns to adopt:**
- Check the internal SDK's pinned dependency versions before installing open-source packages
- When an internal SDK crashes inside a third-party library, the fix is usually in your code (catch the exception, work around it) not in the SDK
- Wrap all external SDK calls in try/except at the boundary — don't let SDK bugs abort your evaluation loop

---

## 12. Evaluation infrastructure is as important as the pipeline itself

**What happened across the session:** Multiple runs overwrote each other (no timestamps), the threshold used wasn't stored in the report, results had to be manually copied to a table, the summary table function was accidentally deleted during an edit, and agent schema metrics were always 0 because the pipeline didn't populate the right field.

**General lesson:** Evaluation infrastructure bugs are insidious because they don't crash loudly — they silently produce wrong or missing data. Treat your eval pipeline with the same rigour as your production pipeline: test it, version its outputs, and make it impossible to lose results.

**Checklist for eval infrastructure:**
- Every run produces a uniquely named output file (timestamp in filename)
- Every hyperparameter used is stored in the output (threshold, model, k values)
- Results are appended to a persistent log (CSV) automatically
- Metrics are computed consistently across all pipeline variants (agent schema metrics must be populated the same way as baseline)
- The eval loop never aborts on a single example failure

---

## Experiment results summary

| Date | Variant | Threshold | Exec Acc | Exact | Recall | Precision | Notes |
|---|---|---|---|---|---|---|---|
| 2026-03-26 | baseline | none | 0.52 | 0.08 | 0.96 | 0.35 | Before fixes — columns=[] bug |
| 2026-03-26 | baseline | 0.8 | 0.56 | 0.14 | 0.63 | 0.56 | Threshold too aggressive, recall collapsed |
| 2026-03-26 | baseline | none | 0.82 | 0.08 | 0.98 | 0.35 | After fixes 1-3 |
| 2026-03-26 | deterministic | none | 0.82 | 0.20 | 1.00 | 0.98 | Refine step achieves near-perfect precision |
| 2026-03-26 | agent | none | 0.76 | 0.06 | 0.00 | 0.00 | Before schema fix — metrics broken |
| 2026-03-26 | agent | none | 0.72 | 0.08 | 1.00* | 0.36 | Turbo makes 0 tool calls, answers from memory |

*recall=1.0 is an artefact of `schema_recall` returning 1.0 when `retrieved_tables=[]` and gold table extraction fails — not a real signal.

---

## Open questions / next experiments

- Does `vertexai::gemini-2.5-flash` actually invoke tools via LangGraph's ReAct format?
- What is the execution accuracy ceiling for single-shot prompting on Spider dev (full set, not just 50)?
- Would adding sample column values to the indexed schema text improve embedding quality enough to make threshold filtering more reliable?
- The 9 remaining deterministic failures include INTERSECT vs UNION errors and the `average` column ambiguity — are these fixable with prompt changes or do they require execution feedback?
- End-to-end few-shot evaluation: does `top_k_examples=0` vs `top_k_examples=3` make a measurable difference on execution accuracy?

---

## 13. Parallelising sync code with asyncio.to_thread

**What happened:** All three pipelines are synchronous and blocking. Running 1034 examples × 3 variants sequentially was slow and a single transient API error aborted the entire run.

**General lesson:** When the bottleneck is network I/O but the code is sync, `asyncio.to_thread` is the right tool. It runs each sync call in the default `ThreadPoolExecutor`, which releases the GIL during I/O and allows true concurrency without rewriting the pipeline layer as async. This is preferable to a full async rewrite when the sync code is complex (e.g. LangGraph pipelines with retry loops) and each unit of work is independent.

Concurrency is bounded with `asyncio.Semaphore(N)` to avoid hammering the API.

---

## 14. Batch I/O beats per-item I/O at scale

**What happened:** The first checkpoint implementation flushed the file after every single example (with a lock). At 1034 examples × 3 pipelines that's ~3000 individual file writes and lock acquisitions.

**General lesson:** Writing to a file (or database) after every item creates unnecessary overhead at scale. A better pattern: gather N results concurrently, write the entire batch in one operation, move to the next batch. This reduces I/O, eliminates lock contention, and bounds memory. The tradeoff is that a crash loses at most one batch — acceptable when batches are small (e.g. 50 examples).

The same principle applies to any append-heavy workload: database inserts, log writes, metric accumulation.

---

## 15. Keep sync and async paths structurally identical

**What happened:** The sync `run_evaluation()` and async `run_evaluation_async()` built result dicts with different keys (`latency_ms` missing from the sync path). This made checkpoint files incompatible with the regular report and would have caused silent metric discrepancies on resume.

**General lesson:** When adding an async variant of an existing function, extract the core logic into shared helpers first. Both paths should call the same functions and produce identical output structures. If they diverge, bugs are hard to trace and data from one path can't be used by the other.

**Pattern used:** `_compute_result_metrics()` (pure function, same output regardless of caller) and `_aggregate()` (builds `EvalReport` from a flat list of dicts). Both sync and async paths call these — the result dicts and final report are guaranteed identical.

---

## 16. A duplicate method definition silently kills tracking

**What happened:** `AgentPipeline` had two `def run()` methods in the same class. Python silently uses the last one. The first method correctly extracted `tool_trace` via `_extract_tool_trace()`; the second omitted it. All 1034 agent results showed `tool_trace: None`, making it look like the model never used tools — but the real cause was dead tracking code.

**General lesson:** Python does not warn on duplicate method definitions. The second definition wins silently. This is especially dangerous in long files where both definitions look plausible. The symptom (a field always being None/empty) looks like a model behaviour issue, not a code bug — which makes it hard to diagnose without reading the class carefully.

**How to catch it:** A linter like `ruff` will flag duplicate method definitions (`E0102` / `F811`). Run `ruff check` as part of CI.

---

## 17. Token usage is available but must be explicitly threaded through

**What happened:** The LLM client returns `usage_metadata` (input/output/total tokens) on every response. `SQLGenerator.generate()` extracted only `response.content` and returned a plain string, silently discarding the usage object. Token counts were never visible in any report.

**General lesson:** LLM SDKs typically attach usage metadata to the response object, not as a separate call. If your wrapper function returns only the content string, usage is gone. The fix is to return a `(content, usage)` tuple (or a dataclass) so callers can choose to record it. Don't discard information at the boundary — it's cheap to carry and expensive to reconstruct.

**Where usage ends up:** `step_timings` in `QueryResponse` — already a `dict[str, Any]`, so it naturally accommodates per-call usage dicts alongside timing values. For the deterministic pipeline with two LLM calls (draft + final), usage is stored under `"draft_usage"` and `"final_usage"` keys separately.

---

## 18. Few-shot examples are already indexed from Spider train set

**What happened:** The analysis recommended adding few-shot examples for INTERSECT/HAVING patterns. Spider `train_spider.json` is already indexed into ChromaDB by `scripts/index_spider.py` and retrieved by all three pipelines. The recommendation was wrong — the examples are there.

**General lesson:** Before recommending "add examples", verify what is already indexed. The retriever may already be finding relevant examples; the real question is whether the retrieved examples are actually helping (end-to-end eval: accuracy with `top_k_examples=0` vs `top_k_examples=3`). If the model ignores them or the retrieval quality is poor, adding more examples won't help — fixing retrieval or prompt formatting will.

**What to actually investigate:** Whether the INTERSECT/HAVING failures are cases where no matching train example was retrieved, or cases where a matching example was retrieved but the model still generated the wrong pattern.
