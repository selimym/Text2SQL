# Text2SQL Failure Taxonomy

This document categorizes all failure modes in the Text2SQL system, defining detection signals, root causes, and repair strategies.

## Overview

Failures occur at three stages:
1. **Retrieval**: schema and few-shot example selection
2. **Generation**: LLM SQL generation quality
3. **Execution**: SQL validation and database execution

The repair loop classifies failures as `retrieval_fault` (missing/wrong context) or `generation_fault` (correct context, wrong logic) and routes to appropriate recovery nodes.

---

## Failure Categories

### 1. Schema Recall Failure

**Name:** `schema_recall_failure`

**Description:**
A required table or column is not retrieved during the schema retrieval phase, causing the LLM to generate SQL that references non-existent schema entities or missing joins.

**Detection Signal:**
- Metric: `schema_recall < 1.0` (not all gold tables in generated SQL appear in retrieved schema docs)
- Flag: `retrieval_guardrail_failed` if no schema docs retrieved at all
- Error observation: validation or execution fails with "table/column not found"

**Typical Cause:**
- Retrieval ranking fails to surface tables relevant to the question
- Question phrasing differs significantly from table/column names in embeddings
- Top-k too small for complex multi-table queries

**Recommended Repair Strategy:**
- Route to `broaden_schema` node (retrieval_fault classification) to fetch more schema docs
- Increase `top_k_schema` parameter for future requests
- Review embedding quality and consider schema augmentation (aliases, synonyms)

---

### 2. Schema Noise Failure

**Name:** `schema_noise_failure`

**Description:**
Irrelevant or tangential schema context is retrieved alongside relevant tables, increasing prompt size and biasing the LLM toward incorrect joins or column selections.

**Detection Signal:**
- Metric: `schema_noise_ratio = 1 - schema_precision` (fraction of retrieved tables not in gold SQL)
- Flag: `schema_docs_truncated` when retrieval docs exceed `max_schema_docs` (guardrail truncates)
- Observation: generated SQL uses wrong table or join path despite validation passing

**Typical Cause:**
- Broad question terms match many unrelated tables
- Embedding model conflates semantically similar tables
- Retrieved docs include transitive FK relationships not needed for the query

**Recommended Repair Strategy:**
- In repair loop, do not broaden schema further; instead re-generate with same docs (generation_fault path)
- Reduce `top_k_schema` for higher-precision retrieval
- Improve question or schema documents with more specificity
- Consider query-schema filtering to exclude obviously irrelevant tables

---

### 3. Few-shot Misguidance

**Name:** `fewshot_misguidance`

**Description:**
Retrieved few-shot examples demonstrate SQL patterns that are superficially similar to the target question but encode the wrong logic, misleading the LLM into replicating an incorrect approach.

**Detection Signal:**
- Metric: `fewshot_table_overlap < 1.0` (retrieved examples use different tables than gold SQL)
- Flag: `example_docs_truncated` if examples exceed `max_example_docs`
- Observation: generated SQL structurally matches a retrieved example but solves the wrong problem

**Typical Cause:**
- Example similarity metric (e.g., embedding distance) is table-agnostic
- Example selection ranks by question similarity, not answer pattern relevance
- Too few examples in retrieval database; closest match is still poorly aligned

**Recommended Repair Strategy:**
- Re-generate with critique emphasizing correct semantic intent (generation_fault path)
- Increase `top_k_examples` to surface better-aligned examples
- Refine example retrieval to weight table overlap more heavily
- Audit example database for confusingly similar but semantically distinct queries

---

### 4. Hallucinated Schema Reference

**Name:** `hallucinated_schema_reference`

**Description:**
The LLM generates SQL referencing tables or columns that do not exist in the database schema, often due to incomplete or misleading schema context, or the LLM inventing plausible-sounding identifiers.

**Detection Signal:**
- Validation error: `"Parse error"` or `"Table/column not found"` during schema validation
- Execution error: `"no such table"` or `"no such column"` from SQLite
- Flag: `sql_guardrail_blocked` if SQL references disallowed keywords (PRAGMA, ATTACH)
- Observation: mentioned_tables in `refine_schema_context_node` contain names not in any schema_docs

**Typical Cause:**
- Schema context incomplete; LLM infers plausible column names
- Question assumes tables by name without explicit schema mention
- LLM confuses aliases or CTEs with real table names

**Recommended Repair Strategy:**
- Classify as `retrieval_fault` if hallucinated table is actually in the database but not retrieved
- Broaden schema retrieval to include the unresolved tables
- If table truly does not exist, return error with suggestion of available tables
- Critique to clarify available schema and correct the query

---

### 5. Wrong Join Path

**Name:** `wrong_join_path`

**Description:**
SQL correctly references all required tables but joins them on the wrong foreign key path, producing semantically incorrect results (cross-product, missing rows, or wrong aggregates).

**Detection Signal:**
- Execution succeeds but result set is wrong (caught in offline evaluation)
- Flag: None specific at runtime; detected post-hoc by result comparison
- Observation: query returns data but fails `result_set_match` against gold result
- Optional flag at runtime: `sql_guardrail_warning` if no LIMIT clause and many rows returned

**Typical Cause:**
- Schema retrieval includes multiple tables but not their relationship metadata
- LLM lacks examples demonstrating the correct FK path in similar scenarios
- Ambiguous schema: multiple FK paths between tables, LLM picks the wrong one

**Recommended Repair Strategy:**
- Critique to explicitly mention correct FK relationships and why the previous join was wrong
- Re-generate with all schema docs still present (generation_fault path)
- Enhance schema docs to include FK cardinality and direction notes
- Improve few-shot examples to cover multi-table joins for this domain

---

### 6. Syntax Error

**Name:** `syntax_error`

**Description:**
The generated SQL cannot be parsed by the SQL validator (sqlglot) due to syntax violations, invalid function calls, or structural malformation.

**Detection Signal:**
- Validation result: `valid=False`, `error="Parse error: ..."`
- Execution skipped; validation short-circuits the pipeline
- Flag: None set by validator, but execution is blocked
- Observation: sqlglot.parse() raises ParseError on generated_sql

**Typical Cause:**
- LLM generates invalid SQL dialect (e.g., MySQL syntax in SQLite context)
- Unclosed parentheses, missing commas, malformed CASE statements
- LLM hallucinates function names not in SQLite (e.g., DATEPART instead of DATE)

**Recommended Repair Strategy:**
- Critique with the exact parse error message and corrected example
- Re-generate with stronger prompt guidance on SQLite syntax
- Provide schema docs with function availability notes
- Do not broaden schema (generation_fault path)

---

### 7. Execution Timeout

**Name:** `execution_timeout`

**Description:**
The query completes syntactic validation but exceeds the execution time budget (default 30 seconds), likely due to full table scans, expensive joins, or unoptimized subqueries.

**Detection Signal:**
- Execution result: `success=False`, `error="Query timed out"`, `error_category="timeout"`
- Latency: `latency_ms >= timeout_seconds * 1000`
- No rows returned; query was terminated mid-execution

**Typical Cause:**
- No WHERE clause to filter large tables
- Cartesian product due to missing join condition or wrong join type
- Subquery or aggregate over entire table without index
- Correct logic but inefficient implementation

**Recommended Repair Strategy:**
- Critique to suggest adding WHERE filters or LIMIT to reduce result set
- Re-generate focusing on query efficiency (generation_fault path)
- Provide schema docs with table size hints and common filter columns
- Consider suggesting simplified version of the query if full query times out

---

### 8. Empty Result

**Name:** `empty_result`

**Description:**
The SQL executes successfully with zero syntax errors but returns an empty result set (no rows). This may be correct (e.g., no matching rows) or incorrect (filters are too restrictive).

**Detection Signal:**
- Execution result: `success=True`, `row_count=0`, `rows=[]`
- Flag: `uncertainty_disclosure` set by `check_output_guardrail` if execution is not successful OR no results
- Observation: offline evaluation compares empty result against non-empty gold result (incorrect)

**Typical Cause:**
- Filters (WHERE clause) too restrictive due to typo or wrong column reference
- Join condition filters out all rows (e.g., wrong FK cardinality assumption)
- Correct logic but no rows match in the current database state
- Question is ambiguous; empty result is a valid interpretation

**Recommended Repair Strategy:**
- At runtime: flag with uncertainty_disclosure and return empty result
- In offline eval: accept empty result only if it matches gold exactly
- Post-execution critique: suggest relaxing filters or checking join conditions
- Provide schema docs with sample data / cardinality hints

---

### 9. Fabricated Answer

**Name:** `fabricated_answer`

**Description:**
Execution fails (validation error, timeout, or runtime error) but the LLM generates a non-empty text answer anyway, inventing data that was never retrieved or computed from the database.

**Detection Signal:**
- Execution metadata: `success=False` (validation failed, timeout, or execution error)
- Response: `answer != ""` (LLM provided text despite failure)
- Flag: `uncertainty_disclosure` set when output guardrail fails
- Observation: audit logs show execution failure but answer contains specific data

**Typical Cause:**
- LLM trained to be helpful and invents plausible-sounding data
- Prompt template does not explicitly forbid answering when execution fails
- RAG pipeline returns few-shot examples with similar questions but different databases
- Hallucination of query results based on question context alone

**Recommended Repair Strategy:**
- Prompt engineering: explicitly instruct LLM to return null/error when execution fails
- Output guardrail: block non-empty answers if execution metadata.success is False
- Include in response a flag indicating execution status
- Post-execution: validate that rows in answer match execution_result.rows exactly

---

### 10. Policy Violation

**Name:** `policy_violation`

**Description:**
The generated SQL attempts to execute non-SELECT statements (INSERT, UPDATE, DELETE, CREATE, ALTER, DROP), violating the read-only policy of the system.

**Detection Signal:**
- Validation result: `valid=False`, `error="Only SELECT allowed, got: Insert|Update|..."`
- Flag: `sql_guardrail_blocked` when SQL contains disallowed DDL/DML keywords
- Observation: SQLValidator detects non-SELECT statement type during parsing

**Typical Cause:**
- Question phrasing ambiguously implies modification (e.g., "add a new user")
- LLM misinterprets the intent or is not prompted to generate SELECT-only queries
- Prompt context includes examples of INSERT/UPDATE/DELETE queries from training data
- Question explicitly asks for schema modification

**Recommended Repair Strategy:**
- Reject immediately; do not attempt repair or retry
- Return error response explicitly stating SELECT-only constraint
- Optionally re-prompt with clarified intent if question is ambiguous
- Input guardrail: pre-filter questions that mention DDL/DML keywords

---

## Repair Flow Integration

The pipeline repair loop uses `fault_category` routing:

```
critique_failure_node (classify as retrieval_fault or generation_fault)
  ├─ retrieval_fault → broaden_schema_node → assemble_prompt → generate_final_sql
  └─ generation_fault → assemble_prompt (reuse schema_docs) → generate_final_sql
```

### Mapping Failures to Fault Categories

| Failure | Fault Category | Repair Action |
|---------|---|---|
| schema_recall_failure | retrieval_fault | Broaden schema, retry generation |
| schema_noise_failure | generation_fault | Re-generate with same schema |
| fewshot_misguidance | generation_fault | Critique with intent clarification, re-generate |
| hallucinated_schema_reference | retrieval_fault (if real) or generation_fault (if false) | Broaden or re-generate |
| wrong_join_path | generation_fault | Critique FK relationships, re-generate |
| syntax_error | generation_fault | Critique parse error, re-generate |
| execution_timeout | generation_fault | Critique efficiency, re-generate with LIMIT suggestion |
| empty_result | generation_fault (if incorrect) | Critique filters, re-generate |
| fabricated_answer | generation_fault | Prompt update to forbid fabrication on failure |
| policy_violation | none (immediate reject) | Return error, do not retry |

---

## Metrics & Observability

### Diagnostic Metrics

- **schema_recall**: Fraction of gold tables in retrieved schema (1.0 = no recall failure)
- **schema_precision**: Fraction of retrieved tables in gold SQL (1.0 = no noise)
- **schema_noise_ratio**: 1 - schema_precision (fraction of irrelevant retrieved tables)
- **fewshot_table_overlap**: Average table overlap between retrieved examples and gold SQL

### Runtime Flags

- `retrieval_guardrail_failed`: No schema docs retrieved or exceeds limits
- `schema_docs_truncated`: More than max_schema_docs retrieved; truncated
- `example_docs_truncated`: More than max_example_docs retrieved; truncated
- `sql_guardrail_blocked`: SQL violates safety policy (disallowed keywords, non-SELECT)
- `sql_guardrail_warning`: SQL passes guardrail but may be risky (no LIMIT, no aggregate)
- `uncertainty_disclosure`: Execution failed or no results; answer may be fabricated

### Execution Result Fields

- `ExecutionResult.success`: Boolean; True if query ran without timeout/error
- `ExecutionResult.error`: Error message (null if success)
- `ExecutionResult.error_category`: "timeout" | "execution_error" | None
- `ExecutionResult.rows`: Result set (empty if no rows or error)
- `ExecutionResult.row_count`: Number of rows returned (0 for empty result)

---

## Best Practices for Debugging

1. **Check flags first**: Flags indicate guardrail violations and truncation
2. **Compare metrics**: High noise or low recall suggests retrieval failure
3. **Trace retry_count**: Multiple retries indicate persistent generation or retrieval issues
4. **Review fault_category**: Classifier's assessment (retrieval vs. generation fault)
5. **Inspect schema_docs**: List of retrieved tables and columns for the query
6. **Compare draft_sql vs. generated_sql**: Identifies if refinement step improved context
7. **Check step_timings**: Execution timeout will have high execute_sql latency

---

## References

- [Pipeline Architecture](architecture.md) — LangGraph topology and node flow
- [Configuration](configuration.md) — Retrieval limits, timeout settings, guardrail thresholds
- [Evaluation Metrics](modules.md#eval-module) — schema_recall, schema_precision definitions
