# Simplified V1 Plan: Deterministic Finance Research Agent with SGLang

## Goal

Build a course-project-scale finance research agent that runs local LLM inference with SGLang, uses MCP servers for financial and web data retrieval, and follows a three-agent workflow:

1. **Researcher agent** gathers information and produces claims with references.
2. **Reviewer agent** verifies those claims against the cited sources and optionally performs limited re-checking.
3. **Formatter agent** takes only the verified claims and writes the final report.

The system is intentionally minimal for v1. It will use:

- one backend application,
- one local SGLang server,
- one SQLite database,
- no cross-run memory,
- no Redis,
- no PostgreSQL,
- no containers,
- no formal observability stack,
- no evaluation framework in the initial build.

SGLang is a good fit because it supports structured outputs and tool calling, which are the two most important capabilities for this design.[1][2][3]

## Model and Hardware

### Model

The system uses **Gemma 4 E4B** (`google/gemma-4-E4B-it`) for all inference.

Key properties relevant to this project:

- **Effective parameters:** 4.5B (8B total with embeddings)
- **Context window:** 128K tokens — sufficient for multi-source research prompts
- **Native function calling:** Yes — critical for tool-calling agents
- **Thinking mode:** Available via `<|think|>` token — can be enabled selectively for the Reviewer
- **Structured output:** Supported through SGLang's constrained decoding
- **License:** Apache 2.0

### Hardware

The system runs on an **Apple M1 Pro with 16GB unified memory**.

The BF16 model weights are approximately 16GB on disk. This leaves no headroom for the KV cache and activations at full precision. Therefore:

- **Use a 4-bit quantized variant** of Gemma 4 E4B. A Q4_K_M GGUF quantization reduces model weight memory to approximately 4–5GB, leaving 10–11GB for the KV cache, system, and the backend process.
- SGLang supports llama.cpp-compatible backends. For Apple Silicon, use SGLang with the Metal backend or run the quantized model via `llama-cpp-python` as a drop-in server behind the same OpenAI-compatible API surface.
- Keep prompts lean. The 128K context window is available in principle, but at Q4 on 16GB, practical safe context is around 32K–64K before memory pressure degrades throughput.
- Think mode (`enable_thinking=True`) should be used only on Reviewer calls where reasoning quality matters more than speed. All other calls use `enable_thinking=False`.

### Sampling Note

The Gemma 4 model card recommends `temperature=1.0, top_p=0.95, top_k=64` for general use. This project overrides that with `temperature=0.0` for all calls to maximize determinism and structured output reliability, which is the correct trade-off for a pipeline that enforces JSON schemas.

## Core Product Definition

The system answers a user’s finance research question by:

- decomposing the question into research angles and subquestions,
- collecting evidence from MCP tools,
- attaching references to every claim,
- independently reviewing the claims,
- generating a final report using only reviewed claims.

The central design principle is simple:

**The first agent is allowed to gather broadly. The second agent is responsible for skepticism and verification only. The third agent writes the final report from only the claims that passed verification.**

## What the System Must Do

The initial version must support these user expectations:

- Accept a finance-related natural-language query.
- Use MCP tools to fetch relevant data and sources.
- Preserve references for everything the Researcher claims.
- Ensure the Reviewer checks those claims before they appear in the final answer.
- Produce a clean final report with references.
- Run entirely with local LLM inference through SGLang.[4][3]

## What the System Will Not Do in V1

To keep the first version manageable, the system will **not** include:

- persistent memory across runs,
- user accounts,
- background job queues,
- distributed services,
- dashboards,
- benchmark/evaluation automation,
- production deployment features,
- complicated ranking or quality-scoring systems,
- support for many simultaneous users.

These can be added later if needed.

## Fixed High-Level Architecture

The v1 architecture has exactly seven parts:

1. **Frontend or CLI**
2. **Backend application**
3. **SGLang inference server**
4. **MCP tool layer**
5. **SQLite storage**
6. **Formatter agent**
7. **Report renderer**

### Frontend or CLI

The system can start as either:

- a simple CLI, or
- a small web UI.

The frontend only needs to:

- accept the user query,
- trigger a run,
- display the final report,
- optionally show sources and intermediate claim-review status.

A CLI is faster to build first. A minimal web UI can be added after the core pipeline works.

### Backend Application

Use a single Python backend application. FastAPI is the preferred choice because it is simple, well suited for structured JSON APIs, and easy to combine with SQLite and local model calls.

The backend owns:

- request handling,
- orchestration,
- calls to SGLang,
- calls to MCP tools,
- source normalization,
- saving artifacts in SQLite,
- final report rendering.

### SGLang Server

Use one local SGLang server for both agents. SGLang should be used through its OpenAI-compatible style API plus structured outputs and tool-calling support.[4][1][2]

SGLang will be the only LLM-serving runtime in the system. There will be no vLLM fallback in v1.

### MCP Tool Layer

The system will connect to these MCP-backed sources:

1. Yahoo Finance MCP
2. FRED MCP
3. Financial Datasets MCP
4. Open Web Search MCP

This stack is already a strong starting set for finance research because Yahoo Finance covers market/company data, FRED covers macro context, Financial Datasets covers richer finance/news data, and Open Web Search covers general discovery and corroboration.[5][6]

### SQLite Storage

Use only one SQLite database in v1.

The database is used only to store the artifacts of the current runs so the pipeline is traceable and debuggable. It is **not** being used as long-term memory or as a complex analytics store.

### Report Renderer

The report renderer converts reviewed claims into a human-readable final report. It should generate markdown first because markdown is simple and easy to debug. HTML rendering can be added later.

## Simplified Workflow

The full system workflow for one query is:

1. Receive user query.
2. Normalize query.
3. Plan research angles and subquestions.
4. Run Researcher agent with MCP tools.
5. Normalize retrieved sources.
6. Build claim list with references.
7. Run Reviewer agent — produces `ClaimReviewSet` only.
8. Optionally perform one retry/research repair loop if needed.
9. Run Formatter agent — takes approved claims and produces `FinalReport`.
10. Save run artifacts to SQLite.
11. Return final report.

This is the only loop allowed in v1:

- Reviewer finds important gaps or unsupported claims.
- Orchestrator sends a structured `RetryInstruction` to the Researcher.
- Researcher gets one additional pass to fill those gaps.
- Reviewer rechecks the full claim set.

No deeper recursive agent workflow is allowed.

## Query Handling

### Input

User input will contain:

- raw query text,
- timestamp,
- optional output style preference.

A simple input model:

```json
{
  "query": "What are the biggest risks for Tesla over the next 12 months?",
  "as_of": "2026-04-11T20:04:00Z",
  "output_style": "memo"
}
```

### Query Normalization

The backend performs light normalization before planning:

- trim whitespace,
- normalize casing where appropriate,
- extract obvious entities like ticker symbols if present,
- preserve the original user query,
- attach current timestamp.

Importantly, the system will **not** force the query into one rigid class like single-stock or macro-only. One finance query may involve multiple dimensions, so the planner should instead produce multiple **research angles** rather than one fixed category.

## Planning Stage

### Planner Purpose

The planner turns the user query into a small structured plan for the Researcher.

The planner should answer:

- What should be investigated?
- Which angles matter?
- Which tools are likely needed?
- What subquestions should the Researcher answer?

### Research Angles

Instead of a single class, the planner outputs a list from a small reusable set of angles:

- `company`
- `macro`
- `news`
- `valuation`
- `comparison`
- `risk`
- `business_quality`

A query can contain any number of these angles.

Example:

```json
{
  "research_angles": ["company", "news", "risk", "macro"],
  "subquestions": [
    "What are the latest company-specific developments?",
    "What do recent financial data sources show?",
    "Are there macro conditions that materially affect this view?",
    "What are the main downside risks over the next 12 months?"
  ],
  "suggested_tools": [
    "financial_datasets",
    "yahoo_finance",
    "fred",
    "open_web_search"
  ]
}
```

### Planner Rules

- The planner must always output structured JSON.[1]
- It must produce 2 to 6 subquestions.
- It must not produce redundant subquestions.
- It must not assume only one type of financial analysis.
- It must include at least one structured-data source when the question involves companies, prices, or financial performance.
- It must include web search when recency or news matters.[5]

## MCP Tool Strategy

### Why MCP Is Used

The LLM should not directly “know” finance facts from memory for current analysis. Instead, the system uses MCP tools to retrieve external information and then reason over that information. This keeps the workflow grounded in data rather than pure parametric recall.

### MCP Tools Used

#### Yahoo Finance MCP
Use for:

- stock price context,
- market cap,
- simple fundamentals,
- recent company snapshot data.

#### FRED MCP
Use for:

- rates,
- inflation,
- unemployment,
- macro time series,
- economic backdrop.[6]

#### Financial Datasets MCP
Use for:

- financial statements,
- richer company financial data,
- news or finance-oriented datasets depending on endpoint availability.[6]

#### Open Web Search MCP
Use for:

- recent news,
- broader discovery,
- corroborating qualitative commentary,
- finding recent developments not easily captured in structured feeds.[5]

### Tool Access Policy

- The Researcher can use all configured MCP tools.
- The Reviewer can also use MCP tools, but only for limited verification or conflict resolution.
- The Reviewer should not do broad research from scratch; it should only re-check material issues.

### Tool Budgets

To keep the system deterministic and bounded:

- Researcher total MCP calls per run: **20**
- Reviewer total MCP calls per run: **6**
- Retry loop allowed: **1**

These numbers are fixed in v1.

## Source Normalization

Every tool result must be normalized into a common `SourceRecord` structure before being passed to the Reviewer or stored in SQLite.

### SourceRecord

```json
{
  "source_id": "src_001",
  "provider": "fred",
  "source_type": "macro_series",
  "title": "Federal Funds Effective Rate",
  "uri": "...",
  "retrieved_at": "2026-04-11T20:04:00Z",
  "published_at": null,
  "entity": null,
  "content_summary": "Latest rate level and recent trend",
  "raw_excerpt": "...",
  "structured_payload": {}
}
```

### Why This Matters

This normalized source format keeps the Reviewer independent from the raw quirks of each MCP tool. The Reviewer only sees a clean, uniform representation of evidence.

## Researcher Agent

### Researcher Goal

The Researcher’s job is to gather relevant evidence and produce claims with explicit references.

It is optimized for **coverage**, not for polished final writing.

### Researcher Inputs

The Researcher receives:

- the original user query,
- the planner output,
- the tool list,
- the current collected sources,
- the structured output schema.

### Researcher Outputs

The Researcher must produce a structured `ResearchResult` object.

```json
{
  "subquestion_answers": [
    {
      "subquestion": "What are the latest company-specific developments?",
      "summary": "The company faced margin pressure and weaker delivery growth.",
      "source_ids": ["src_001", "src_002"]
    }
  ],
  "claims": [
    {
      "claim_id": "clm_001",
      "text": "Recent results indicate slower growth than the prior year.",
      "source_ids": ["src_001", "src_003"],
      "support_type": "direct"
    }
  ],
  "gaps": [
    "No direct primary-source confirmation found for one recent media claim."
  ]
}
```

### Researcher Rules

- Every claim must include at least one source id.
- Numeric claims should include at least two sources when possible.
- Claims must be factual and evidence-based, not just speculative opinions.
- If the Researcher is unsure, it must record a gap rather than invent a claim.
- Contradictory findings should appear as separate claims or gaps, not merged away.
- The Researcher does not write the final user-facing report.

### Researcher Retrieval Pattern

For each subquestion:

1. Pick the most relevant tool.
2. Retrieve one or more sources.
3. Normalize the source outputs.
4. Check whether enough evidence exists.
5. Continue until the subquestion is sufficiently supported or budget is exhausted.
6. Move to the next subquestion.

### Researcher Stopping Conditions

Stop research when any of these is true:

- all subquestions have at least minimal support,
- tool budget is exhausted,
- two consecutive retrieval actions add no meaningful new evidence,
- the retry loop has already been used.

## Reviewer Agent

### Reviewer Goal

The Reviewer verifies the Researcher’s claims. It produces a `ClaimReviewSet` only — it does not write the final report. That responsibility belongs to the Formatter agent.

It is optimized for:

- correctness,
- evidence discipline,
- conflict detection.

### Reviewer Inputs

The Reviewer receives:

- original user query,
- planner output,
- normalized sources,
- Researcher claims,
- gaps identified by the Researcher.

### Reviewer Outputs

The Reviewer produces one structured output: `ClaimReviewSet`.

```json
{
  "claim_reviews": [
    {
      "claim_id": "clm_001",
      "verdict": "verified",
      "notes": "Supported by recent financial data and corroborating source.",
      "final_source_ids": ["src_001", "src_003"],
      "needs_recheck": false
    }
  ],
  "global_decision": {
    "needs_retry": false,
    "retry_focus_subquestions": [],
    "unsupported_claim_ids": []
  }
}
```

### Reviewer Verdict Options

Each claim must receive one of these verdicts:

- `verified`
- `partially_verified`
- `unsupported`
- `contradicted`

### Reviewer Rules

- A claim is `verified` if the cited evidence directly supports it.
- A claim is `partially_verified` if the direction is right but the wording is too strong.
- A claim is `unsupported` if the cited evidence does not actually support it.
- A claim is `contradicted` if better or clearer evidence points the other way.
- Only `verified` and `partially_verified` claims are passed to the Formatter.
- `unsupported` and `contradicted` claims must not appear in the final report body.
- If the missing evidence is important and recoverable, the Reviewer sets `needs_retry: true` and populates `retry_focus_subquestions`.

### Reviewer Recheck Policy

The Reviewer may perform limited extra MCP calls only when:

- a claim is important to the final answer,
- the current evidence is close but insufficient,
- two sources materially conflict,
- freshness matters and one source looks outdated.

The Reviewer should not restart the entire research process.

### Thinking Mode

The Reviewer call uses `enable_thinking=True`. Verification requires careful reasoning about whether evidence actually supports a claim. The thinking trace is discarded after the call — only the `ClaimReviewSet` JSON is retained.

## Retry Instruction Payload

When the Reviewer sets `needs_retry: true`, the orchestrator builds a structured `RetryInstruction` and sends it to the Researcher for one additional pass.

### RetryInstruction Schema

```json
{
  "retry_instruction": {
    "retry_reason": "Two key risk claims are unsupported. Primary source data is missing for the valuation and macro subquestions.",
    "focus_subquestions": [
      "What does the latest financial data show about valuation multiples?",
      "Are there macro conditions that materially affect the near-term outlook?"
    ],
    "unsupported_claims": [
      {
        "claim_id": "clm_003",
        "claim_text": "Current valuation multiples are at a five-year high.",
        "rejection_reason": "No structured data source was cited. Only a web search snippet with no primary data."
      }
    ],
    "gaps_to_fill": [
      "No direct primary-source confirmation found for one recent media claim."
    ],
    "already_retrieved_source_ids": ["src_001", "src_002", "src_003"],
    "suggested_tools": ["yahoo_finance", "financial_datasets"],
    "remaining_tool_budget": 6
  }
}
```

### RetryInstruction Fields

| Field | Purpose |
|-------|---------|
| `retry_reason` | Short plain-language explanation of why a retry is needed |
| `focus_subquestions` | The specific subquestions the Researcher must focus on |
| `unsupported_claims` | Claims that were rejected, with the reason, so the Researcher knows what evidence is missing |
| `gaps_to_fill` | Gaps identified in the original pass that are still unresolved |
| `already_retrieved_source_ids` | Source ids already in the database — the Researcher must not re-fetch these |
| `suggested_tools` | Tools most likely to fill the missing evidence |
| `remaining_tool_budget` | How many additional MCP calls the Researcher may make in this retry pass |

### Retry Rules

- The Researcher only works on the `focus_subquestions` and `unsupported_claims` listed in the instruction. It does not redo the full research pass.
- The Researcher must not re-fetch source ids in `already_retrieved_source_ids`.
- The retry budget is fixed at 6 additional MCP calls.
- After the retry, the Reviewer rechecks all claims — both original and newly added.
- If claims are still unsupported after the retry, they are excluded from the final report. No second retry is allowed.

## Formatter Agent

### Formatter Goal

The Formatter receives only the approved claims from the Reviewer and writes the final human-readable report. It has no access to MCP tools and performs no verification.

This separation keeps the writing task clean: the Formatter never sees unsupported claims and cannot accidentally include them.

### Formatter Inputs

The Formatter receives:

- original user query,
- as-of timestamp,
- output style preference (`memo`, `brief`, or `full`),
- only the `verified` and `partially_verified` claims with their resolved source references,
- the list of unverified items and gaps to place in the caveats section.

### Formatter Outputs

The Formatter produces one structured output: `FinalReport`.

```json
{
  "title": "Tesla risk research report",
  "as_of": "2026-04-11T20:04:00Z",
  "output_style": "memo",
  "executive_summary": [
    "The main near-term risks are slower growth, margin pressure, and externally driven demand sensitivity."
  ],
  "sections": [
    {
      "heading": "Key risks",
      "paragraphs": [
        "..."
      ]
    }
  ],
  "unverified_items": [
    "One recent commentary item could not be independently confirmed."
  ],
  "reference_source_ids": ["src_001", "src_002", "src_003"]
}
```

### Output Style Definitions

| Style | Description |
|-------|-------------|
| `memo` | One-page format. Executive summary plus key points. Ideal for quick reads. |
| `brief` | Short summary only. Two to four sentences. No section headings. |
| `full` | Complete report with all required sections, detailed paragraphs, and full reference list. |

### Formatter Rules

- The Formatter may only reference source ids passed to it — it cannot introduce new claims.
- Every section must be grounded in at least one approved claim.
- Unverified items appear in the dedicated caveats section only, never in the main analysis.
- The Formatter uses `enable_thinking=False` — this is a structured writing call, not a reasoning call.

## Final Report Generation

### Report Philosophy

The final report is built from reviewed claims, not from raw unfiltered model prose.

That means:

- the structure of the report is controlled by the `FinalReport` schema,
- the content comes exclusively from claims that passed the Reviewer,
- references are derived from claim-source links, not invented inline.

### Required Report Sections

Every final report must include:

1. Title
2. As-of timestamp
3. Executive summary
4. Main analysis
5. Risks / caveats
6. Unverified or conflicting items
7. References

### Reference Handling

References are not invented in free text by the model. Instead:

- the Formatter outputs source ids,
- the backend resolves those ids into titles, URLs, and provider names from the `sources` table,
- the renderer formats them at the bottom of the report.

This keeps citation handling deterministic and auditable.

## Configuration Management

All runtime settings are stored in a `.env` file at the project root. The backend reads this file using `python-dotenv` at startup. A `.env.example` file with placeholder values is committed to the repository; the actual `.env` is gitignored.

### .env Fields

```dotenv
# SGLang / inference server
SGLANG_BASE_URL=http://127.0.0.1:30000
SGLANG_MODEL_ID=google/gemma-4-E4B-it-q4
SGLANG_CONTEXT_LIMIT=32768

# MCP server endpoints
YAHOO_FINANCE_MCP_URL=http://127.0.0.1:8001
FRED_MCP_URL=http://127.0.0.1:8002
FINANCIAL_DATASETS_MCP_URL=http://127.0.0.1:8003
OPEN_WEB_SEARCH_MCP_URL=http://127.0.0.1:8004

# API keys for MCP tools (if required)
FRED_API_KEY=
FINANCIAL_DATASETS_API_KEY=

# Storage
SQLITE_PATH=data/app.db

# Tool budgets
RESEARCHER_TOOL_BUDGET=20
REVIEWER_TOOL_BUDGET=6
RETRY_TOOL_BUDGET=6

# Logging
LOG_LEVEL=INFO
```

### Config Module

A `config.py` module reads the `.env` file and exposes a single `Settings` object imported throughout the backend. No module hardcodes any URL, key, or budget value directly.

## Source ID Generation

Every `SourceRecord` gets a stable, unique id generated at normalization time using this format:

```
{run_id_prefix}-{provider_prefix}-{uuid4_short}
```

Where:

- `run_id_prefix` is the first 8 characters of the run’s UUID.
- `provider_prefix` is a fixed 4-character code per provider: `yhoo`, `fred`, `find`, `webs`.
- `uuid4_short` is the first 8 characters of a new UUID4 generated at source creation time.

Example: `a3f9b12c-yhoo-d7e4c091`

This format makes source ids:

- globally unique within and across runs,
- visually traceable to their provider and run without querying the database,
- safe to use as foreign keys in the `claims` and `reviews` tables.

The id is generated in `source_normalizer.py` at the point of normalization, before the record is written to SQLite. Once assigned, source ids are immutable.

## Determinism Rules

The system is intended to be deterministic enough for a course project demonstration.

### Required Determinism Controls

- temperature = 0.0 for all model calls,
- fixed prompts,
- fixed JSON output schemas,
- stable ordering of sources before passing them to the model,
- fixed tool budget,
- one retry maximum,
- identical prompt structure for repeated runs.

### Why Structured Output Matters

SGLang’s structured output support is important because it lets the agents return schema-constrained JSON instead of messy free-form outputs.[1][7]

### Why Tool Calling Matters

SGLang’s tool-calling support is important because the Researcher and Reviewer agents need controlled access to MCP tools without requiring brittle manual parsing of tool instructions.[2]

## SGLang Usage Plan

### Serving Setup

Run one local SGLang server and connect to it from the backend using the OpenAI-compatible API surface. The backend communicates through `sglang_client.py`, which wraps the HTTP client and applies the fixed per-call profiles below.

### Model Usage

Use Gemma 4 E4B (`google/gemma-4-E4B-it`) in 4-bit quantized form for all three agents. One model, one server. This simplifies debugging and keeps memory usage predictable on the M1 Pro 16GB machine.

### SGLang Profiles

Use these fixed settings per call type. All calls use `temperature=0.0` regardless of the model card’s general recommendation, because determinism and schema compliance are the priority here.

#### Planner call
- temperature: 0.0
- max_tokens: 1000
- enable_thinking: False
- structured output: enabled
- tool calls: disabled

#### Researcher call
- temperature: 0.0
- max_tokens: 2200
- enable_thinking: False
- tool calls: enabled
- structured output for final response: enabled

#### Reviewer call
- temperature: 0.0
- max_tokens: 2400
- enable_thinking: True
- tool calls: enabled (limited recheck budget)
- structured output: enabled
- note: thinking traces are stripped before passing output downstream

#### Formatter call
- temperature: 0.0
- max_tokens: 2000
- enable_thinking: False
- tool calls: disabled
- structured output: enabled

These are initial defaults and can be adjusted during testing.

## Minimal Storage Design

Use only one SQLite database file.

### Why SQLite Is Enough

For this project, SQLite is enough because:

- there is no multi-user concurrency requirement,
- there is no distributed system,
- there is no long-term memory requirement,
- the stored data is only for current or recent runs,
- it makes setup dramatically simpler.

### Tables

Use only these five tables:

#### `runs`
Stores one row per user query.

Fields:
- `run_id`
- `query_text`
- `created_at`
- `status`

#### `sources`
Stores normalized source records for each run.

Fields:
- `source_id`
- `run_id`
- `provider`
- `source_type`
- `title`
- `uri`
- `retrieved_at`
- `content_summary`
- `raw_excerpt`
- `structured_payload_json`

#### `claims`
Stores Researcher claims.

Fields:
- `claim_id`
- `run_id`
- `claim_text`
- `source_ids_json`
- `support_type`

#### `reviews`
Stores Reviewer verdicts.

Fields:
- `review_id`
- `run_id`
- `claim_id`
- `verdict`
- `notes`
- `final_source_ids_json`

#### `reports`
Stores the final report.

Fields:
- `report_id`
- `run_id`
- `title`
- `report_markdown`
- `created_at`

That is the full storage system for v1.

## Backend Modules

Organize the code into a simple structure like this:

```text
finance-research-agent/
  app/
    main.py              # FastAPI app entry point
    api.py               # Route definitions
    config.py            # Reads .env, exposes Settings object
    orchestrator.py      # Pipeline coordination
    planner.py           # Planner agent call
    researcher.py        # Researcher agent call
    reviewer.py          # Reviewer agent call (verification only)
    formatter.py         # Formatter agent call (report writing)
    renderer.py          # Converts FinalReport JSON to markdown
    db.py                # SQLite setup and queries
    models.py            # SQLAlchemy or raw sqlite3 table definitions
    schemas.py           # Pydantic schemas for all structured outputs
    sglang_client.py     # SGLang HTTP client wrapper
    mcp_client.py        # MCP tool call wrapper
    source_normalizer.py # Normalizes raw MCP responses to SourceRecord
  data/
    app.db
  prompts/
    planner.txt
    researcher.txt
    reviewer.txt
    formatter.txt
  .env.example           # Committed template with placeholder values
  .env                   # Gitignored, contains real keys and URLs
  requirements.txt
```

This is intentionally small and easy to reason about.

## Orchestrator Logic

The orchestrator is a normal Python module, not a separate service.

Its responsibilities are:

- receive the request,
- call planner,
- run Researcher,
- save sources and claims,
- run Reviewer (verification only),
- decide whether one retry is needed,
- if retry: build RetryInstruction, run Researcher again, rerun Reviewer,
- run Formatter with approved claims,
- run renderer,
- save final report,
- return result.

### Retry Logic

Only one retry loop is allowed.

Pseudo-flow:

1. Planner runs.
2. Researcher gathers evidence.
3. Reviewer checks claims — produces `ClaimReviewSet`.
4. If `needs_retry: true` and retry not yet used:
   a. Orchestrator builds `RetryInstruction` from the `ClaimReviewSet`.
   b. Researcher receives `RetryInstruction` and runs a targeted second pass.
   c. New sources are saved. New claims are merged into the claim set.
   d. Reviewer rechecks the full claim set (original + new).
5. Orchestrator filters to only `verified` and `partially_verified` claims.
6. Formatter receives the approved claims and writes `FinalReport`.
7. Renderer converts `FinalReport` to markdown.
8. Artifacts are saved to SQLite.

## Prompt Design

### Planner Prompt

The planner prompt should tell the model to:

- identify relevant research angles,
- generate subquestions,
- suggest tool usage,
- output only valid JSON.

### Researcher Prompt

The researcher prompt should tell the model to:

- gather evidence before claiming anything,
- cite source ids for every claim,
- record gaps instead of guessing,
- keep claims factual,
- output only valid JSON.

### Reviewer Prompt

The reviewer prompt should tell the model to:

- verify each claim strictly against its cited sources,
- reject unsupported or exaggerated claims,
- use limited re-checking when needed,
- not write any part of the final report — only produce the `ClaimReviewSet`,
- output only valid JSON.

### Formatter Prompt

The formatter prompt should tell the model to:

- organize only the provided approved claims into the report structure,
- match the requested output style (`memo`, `brief`, or `full`),
- place any unverified items in the caveats section exclusively,
- not introduce any new claims or sources beyond what was provided,
- output only valid JSON.

## Development Phases

### Phase 1: Skeleton

Build:

- FastAPI or CLI entry point,
- SQLite setup,
- SGLang client,
- MCP client wrapper,
- empty orchestrator.

Goal:
A query can pass through a mock pipeline.

### Phase 2: Tool Integration

Build:

- Yahoo Finance MCP integration,
- FRED MCP integration,
- Financial Datasets MCP integration,
- Open Web Search MCP integration,
- source normalization.

Goal:
The backend can fetch and normalize real sources.

### Phase 3: Planner + Researcher

Build:

- planner prompt and schema,
- researcher prompt and schema,
- claim generation with source ids.

Goal:
The system produces claims with references.

### Phase 4: Reviewer + Formatter + Final Report

Build:

- reviewer prompt and schema,
- verdict logic,
- formatter prompt and schema,
- retry instruction builder,
- final markdown report generation.

Goal:
The system produces a reviewed, formatted final report. Reviewer handles verification only. Formatter handles writing only.

### Phase 5: Polish

Build:

- one retry loop,
- cleaner UI or CLI formatting,
- source inspection view,
- better error handling.

Goal:
The project is demo-ready.

## Failure Handling

### Tool Failure

If a tool call fails:

- retry once if the failure is clearly transient,
- otherwise log the failure inside the run,
- continue if enough evidence remains,
- surface the limitation if it affects the final answer.

### Model Output Failure

If SGLang returns invalid structured output:

- retry the same stage once,
- pass a short validation error message,
- if it still fails, stop the run and surface an error.

### Weak Evidence Case

If the system cannot verify enough claims:

- still return a limited report,
- clearly mark uncertainty,
- keep unsupported items outside the main analysis.

## Acceptance Criteria for V1

The first version is successful when:

- a user can ask a finance research question,
- the Researcher gathers sources from MCP tools,
- every claim has a stable source id,
- the Reviewer produces a `ClaimReviewSet` that clearly separates verified from unsupported claims,
- the Formatter produces a `FinalReport` that contains only verified and partially-verified claims,
- the retry loop fires correctly when the Reviewer flags unsupported claims,
- the whole system runs locally using Gemma 4 E4B (Q4) through SGLang on the M1 Pro,
- all settings come from `.env` and no values are hardcoded,
- the codebase remains simple enough to explain in a course project demo.[4][1][2]

## Example End-to-End Scenario

User query:

> What are the main risks for Nvidia over the next 12 months?

Expected flow:

1. Planner outputs angles like `company`, `news`, `risk`, `macro`.
2. Researcher queries structured company data and recent web/news sources.
3. Researcher emits claims like demand concentration risk, valuation sensitivity, or macro-sensitive capex risks, each with source ids.
4. Reviewer checks whether each claim is actually supported.
5. Unsupported claims are removed or isolated.
6. Final report is produced with summary, main risks, caveats, and references.

## Final Design Summary

This v1 system is a small, deterministic, three-agent finance research pipeline built for a course project. It uses Gemma 4 E4B (4-bit quantized) served through SGLang on an M1 Pro Mac for all local inference. MCP servers provide external finance and web data. One Python backend handles orchestration, and one SQLite database stores all run artifacts. The Researcher gathers evidence and cites it. The Reviewer verifies the evidence and produces a structured claim review. The Formatter takes only the approved claims and writes the final report. Configuration is managed through a `.env` file. Everything else is intentionally deferred until the core workflow works reliably.