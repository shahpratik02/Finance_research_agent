# Finance Research Agent

A local LLM-powered finance research pipeline that takes a natural-language question about markets, companies, or economics and produces a sourced, reviewed, markdown report — fully offline-capable with no cloud LLM dependency.

The agent decomposes a query into sub-questions, gathers evidence from four specialised MCP (Model Context Protocol) tool servers and optionally from user-supplied documents (RAG), verifies each claim against all collected sources, and assembles a structured report with references.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Agent](#running-the-agent)
5. [Accessing Outputs](#accessing-outputs)
6. [Agent Architecture](#agent-architecture)
7. [MCP Tool Servers](#mcp-tool-servers)
8. [API Reference](#api-reference)

---

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Runtime for the app and all MCP servers |
| **Ollama** | Latest | Local LLM inference (OpenAI-compatible API) |
| **curl** | Any | Health checks and API interaction |

### Ollama Setup

Install Ollama from [ollama.com](https://ollama.com) and pull a model:

```bash
# Pull the default model (Gemma 4 — 4-bit quantised, ~5 GB)
ollama pull gemma4:e4b-it-q4_k_m

# Start Ollama with a 16K context window
OLLAMA_CONTEXT_LENGTH=16384 ollama serve

# Verify it's running
curl http://localhost:11434/v1/models
```

Any OpenAI-compatible inference server works (SGLang, vLLM, etc.) — just set `SGLANG_BASE_URL` and `SGLANG_MODEL_ID` accordingly.

### API Keys

Two optional (but recommended) API keys unlock additional data sources:

| Key | Source | Free Tier |
|---|---|---|
| `FRED_API_KEY` | [FRED (Federal Reserve)](https://fred.stlouisfed.org/docs/api/api_key.html) | Yes — request at the link |
| `FINANCIAL_DATASETS_API_KEY` | [Financial Datasets](https://financialdatasets.ai) | Yes — sign up at the link |

The Yahoo Finance and Open Web Search servers require **no API keys**.

---

## Installation

```bash
# Clone the repo
git clone <repo-url> && cd Finance_research_agent

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies (app + MCP servers share one venv)
pip install -r requirements.txt
```

MCP servers have their own `requirements.txt` files under `mcp_servers/*/`, but all packages are already included in the root `requirements.txt`.

---

## Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_BASE_URL` | *(required)* | LLM inference URL, e.g. `http://localhost:11434/v1` |
| `SGLANG_MODEL_ID` | *(required)* | Model name, e.g. `gemma4:e4b-it-q4_k_m` |
| `SGLANG_CONTEXT_LIMIT` | `32768` | Max context window tokens. Set to match your model (e.g. `16384` for smaller models) |
| `YAHOO_FINANCE_MCP_URL` | `http://127.0.0.1:8001` | Yahoo Finance MCP server |
| `FRED_MCP_URL` | `http://127.0.0.1:8002` | FRED MCP server |
| `FINANCIAL_DATASETS_MCP_URL` | `http://127.0.0.1:8003` | Financial Datasets MCP server |
| `OPEN_WEB_SEARCH_MCP_URL` | `http://127.0.0.1:8004` | Open Web Search MCP server |
| `FRED_API_KEY` | *(optional)* | FRED API key |
| `FINANCIAL_DATASETS_API_KEY` | *(optional)* | Financial Datasets API key |
| `SQLITE_PATH` | `data/app.db` | Path to the SQLite database file |
| `RESEARCHER_TOOL_BUDGET` | `20` | Max tool calls during the research phase |
| `REVIEWER_TOOL_BUDGET` | `6` | Max tool calls during the review phase |
| `RETRY_TOOL_BUDGET` | `6` | Max tool calls during a retry research pass |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `DEBUG_TRACE` | `false` | Enable detailed debug trace logging (see [Debug Trace](#debug-trace)) |

### Example `.env`

```env
SGLANG_BASE_URL=http://localhost:11434/v1
SGLANG_MODEL_ID=gemma4:e4b-it-q4_k_m
SGLANG_CONTEXT_LIMIT=16384

FRED_API_KEY=your_fred_key_here
FINANCIAL_DATASETS_API_KEY=your_financial_datasets_key_here

SQLITE_PATH=data/app.db
RESEARCHER_TOOL_BUDGET=8
REVIEWER_TOOL_BUDGET=3
RETRY_TOOL_BUDGET=4

LOG_LEVEL=INFO
DEBUG_TRACE=true
```

---

## Running the Agent

### 1. Start MCP Servers

```bash
chmod +x start_mcp_servers.sh stop_mcp_servers.sh
./start_mcp_servers.sh
```

This starts all four MCP tool servers in the background, waits for warm-up (~15 seconds), and runs health checks. Logs are written to `logs/mcp_servers/`. PIDs are stored in `.pid` files for graceful shutdown.

To stop the servers:

```bash
./stop_mcp_servers.sh
```

### 2. Start the FastAPI App

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On startup, the app initialises the SQLite database (creates tables if they don't exist) and is ready to accept requests.

### 3. Run a Research Query

```bash
curl -s -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current state of Apple stock and its outlook?"}' \
  | python3 -m json.tool
```

The pipeline runs synchronously and returns the full report in the response. Depending on the model speed and number of tool calls, a typical run takes 1–5 minutes.

### What Happens Under the Hood (`app/main.py` → `app/orchestrator.py`)

The `POST /research` endpoint calls `run_pipeline()` in the orchestrator, which coordinates the entire multi-phase pipeline:

1. **Cleanup** — Deletes all previous runs from the database (single-run design).
2. **Plan** — The planner LLM call decomposes the query into research angles, sub-questions, and suggested tools.
3. **RAG** *(optional)* — If the request includes `documents` (inline text) and/or `documents_folder` (path to files), the pipeline retrieves chunks from that corpus and runs a structured extraction pass (`app/rag.py`). Output is merged with MCP research when RAG is only partial, or when the query clearly needs **live market data** (e.g. current share price, “current state of … stock”) — in those cases the MCP researcher **always** runs as well, even if RAG labelled its answer `complete`, so PDFs and tools are combined.
4. **Research (MCP)** — When RAG is off, incomplete, or supplemented for live data: the researcher enters a tool-calling loop over MCP tools, normalises results into `SourceRecord` objects, and produces `Claim` rows with source references. Claims from RAG and MCP are merged before review.
5. **Review** — The reviewer sees **every** source (RAG `rag_document` chunks plus MCP rows). For user-document chunks it receives the **full** chunk text in its prompt (not a short head), so claims tied to PDF excerpts can be verified. It assigns a verdict per claim (`verified`, `partially_verified`, `unsupported`, `contradicted`) and may make a small number of extra MCP calls for re-checks.
6. **Retry** *(conditional)* — If the reviewer flags too many unsupported claims, the researcher runs a second focused pass with a smaller tool budget, then the reviewer re-evaluates.
7. **Format** — The formatter builds a structured `FinalReport` from only the approved (verified + partially verified) claims. If **no** claims are approved, it emits a minimal report: caveats list **claim ids and reviewer notes** only (not the full rejected claim text), so unsupported numbers are not echoed as if they were vetted facts.
8. **Render** — The renderer converts the `FinalReport` into markdown, resolving source IDs to full metadata (title, provider, URL).
9. **Persist** — The report is saved to both the SQLite database and as a markdown file at `reports/{run_id}.md`.

---

## Accessing Outputs

The agent produces outputs in three places:

### 1. Markdown Report File — `reports/{run_id}.md`

After each successful run, a standalone markdown report is saved to the `reports/` directory, named by the run ID. Open it directly in any markdown viewer.

```bash
ls reports/
cat reports/<run_id>.md
```

### 2. SQLite Database — `data/app.db`

The database stores the full pipeline state across five tables:

| Table | Contents |
|---|---|
| `runs` | Run metadata (ID, query, status, timestamps) |
| `sources` | All retrieved source records (provider, tool, title, URI, content, raw excerpts) |
| `claims` | Researcher claims with source references and support type |
| `reviews` | Reviewer verdicts for each claim (verdict, notes, final sources) |
| `reports` | Final rendered markdown report |

#### Query the database directly:

```bash
# Get the latest report markdown
.venv/bin/python -c "
from app.db import init_db, get_report_for_run, list_all_run_ids
init_db()
runs = list_all_run_ids()
if runs:
    report = get_report_for_run(runs[-1])
    if report:
        print(report.report_markdown)
"
```

```bash
# List all claims and their review verdicts
.venv/bin/python -c "
from app.db import init_db, list_all_run_ids, get_claims_for_run, get_reviews_for_run
init_db()
for rid in list_all_run_ids():
    for c in get_claims_for_run(rid):
        print(f'{c.claim_id}: {c.claim_text}')
    for r in get_reviews_for_run(rid):
        print(f'  -> {r.claim_id}: {r.verdict} — {r.notes}')
"
```

#### Or use the REST API:

```bash
# Get full run details (run + claims + reviews + report)
curl -s http://127.0.0.1:8000/runs/<run_id> | python3 -m json.tool

# Get all sources for a run
curl -s http://127.0.0.1:8000/runs/<run_id>/sources | python3 -m json.tool

# Delete a run
curl -s -X DELETE http://127.0.0.1:8000/runs/<run_id>
```

### 3. Debug Trace — `logs/debug_trace.log`

When `DEBUG_TRACE=true` is set in `.env`, the agent logs a detailed trace of every LLM request/response, tool call, tool result, and phase transition to `logs/debug_trace.log`. This is invaluable for debugging prompt issues, inspecting raw model outputs, or understanding the tool-calling sequence.

The trace log includes:

- **`NEW RUN`** — Run ID and query
- **`PHASE`** — Phase transitions (PLANNER, RAG, RESEARCHER, REVIEWER, RETRY, FORMATTER, RENDERER)
- **`LLM_REQUEST`** — Full messages sent to the model, schema name, tool definitions
- **`LLM_RESPONSE`** — Raw model output, tool calls, parsed structured data
- **`TOOL_CALL`** — MCP tool name, arguments, target provider
- **`TOOL_RESULT`** — Raw result from MCP server (truncated to keep log manageable)

The log appends across runs (not overwritten), so you can compare multiple runs.

To enable/disable: set `DEBUG_TRACE=true` or `DEBUG_TRACE=false` in `.env` and restart the app.

---

## Agent Architecture

```
┌─────────────────────────────────────────────────┐
│                  FastAPI App                     │
│               (app/main.py)                      │
│                    │                             │
│            POST /research                        │
│                    │                             │
│           ┌────────▼────────┐                    │
│           │   Orchestrator  │ (app/orchestrator)  │
│           │   run_pipeline()│                    │
│           └────────┬────────┘                    │
│                    │                             │
│    ┌───────────────┼───────────────┐             │
│    ▼               ▼               ▼             │
│ ┌──────┐    ┌───────────┐    ┌──────────┐       │
│ │Planner│    │Researcher │    │ Reviewer │       │
│ └──┬───┘    └─────┬─────┘    └────┬─────┘       │
│    │              │               │              │
│    │         ┌────▼────┐          │              │
│    │         │MCP Client│─────────┤              │
│    │         └────┬────┘          │              │
│    │              │               │              │
│    ▼              ▼               ▼              │
│ ┌──────┐    ┌──────────┐    ┌──────────┐        │
│ │Format│    │  Render   │    │   DB     │        │
│ └──────┘    └──────────┘    └──────────┘        │
└─────────────────────────────────────────────────┘
         │                │
    ┌────▼────┐     ┌─────▼─────┐
    │ SGLang  │     │ 4 MCP     │
    │ Client  │     │ Servers   │
    │(OpenAI) │     │ (FastAPI) │
    └─────────┘     └───────────┘
```

### Pipeline Phases

| Phase | Module | LLM Call | Tools | Output |
|---|---|---|---|---|
| **Plan** | `app/planner.py` | 1 call (temp=0) | None | `PlannerOutput` — angles, sub-questions, suggested tools |
| **RAG** *(optional)* | `app/rag.py` | 1–2 calls (temp=0) | Embeddings / Chroma / lexical retrieval | `ResearchResult` fragment + `rag_document` `SourceRecord`s |
| **Research** | `app/researcher.py` | Multi-turn loop | Up to `RESEARCHER_TOOL_BUDGET` | `ResearchResult` — claims with source IDs, gaps (merged with RAG when both run) |
| **Review** | `app/reviewer.py` | 1–2 calls (thinking enabled) | Up to `REVIEWER_TOOL_BUDGET` | `ClaimReviewSet` — verdicts per claim, retry decision |
| **Retry** *(optional)* | `app/researcher.py` | Multi-turn loop | Up to `RETRY_TOOL_BUDGET` | Additional claims + sources |
| **Format** | `app/formatter.py` | 1 call (temp=0) | None | `FinalReport` — structured report from approved claims |
| **Render** | `app/renderer.py` | None | None | Markdown string with resolved references |

### Key Modules

| Module | Purpose |
|---|---|
| `app/config.py` | Settings singleton, reads all env vars |
| `app/schemas.py` | All Pydantic models — `PlannerOutput`, `Claim`, `ClaimReview`, `FinalReport`, etc. |
| `app/models.py` | Database row models — `Run`, `Source`, `Claim`, `Review`, `Report` |
| `app/db.py` | SQLite persistence layer (WAL mode), all CRUD operations |
| `app/llm_client.py` | OpenAI-compatible LLM client with structured output, tool calling, thinking, and optional embeddings |
| `app/rag.py` | Optional document RAG: chunking, retrieval, structured claim extraction |
| `app/rag_corpus.py`, `app/rag_vector_store.py` | Document corpus paths and optional Chroma vector retrieval |
| `app/mcp_client.py` | Routes tool calls to the correct MCP server, handles retries and errors |
| `app/source_normalizer.py` | Transforms raw MCP responses into normalised `SourceRecord` objects |
| `app/trace.py` | Debug trace logger (enabled via `DEBUG_TRACE` env var) |
| `prompts/*.txt` | System prompt templates for each LLM phase |

### LLM Client (`app/llm_client.py`)

The LLM client wraps the OpenAI Python SDK and supports:

- **Structured JSON output** — Pydantic schemas are passed as `response_format` for typed parsing
- **Tool calling** — OpenAI function-calling format; tools are passed as a list of JSON schemas
- **Thinking mode** — For the reviewer, prepends "Think step by step" and extracts `<think>...</think>` traces
- **Context limit** — Passed via `extra_body` for Ollama's `num_ctx` parameter
- **JSON extraction** — Strips markdown fences and finds `{...}` blocks in raw model output

Each phase uses a `CallProfile` with fixed temperature (0.0) and per-phase `max_tokens`.

### Source Normalisation (`app/source_normalizer.py`)

Raw MCP tool responses are heterogeneous (each server returns different JSON shapes). The normaliser extracts consistent fields:

- **source_id** — Deterministic: `{run_id[:8]}-{provider_prefix}-{uuid[:8]}`
- **title, uri, entity** — Extracted per provider
- **content_summary** — Short text summary for the LLM
- **raw_excerpt** — Short verbatim slice for storage and UI; length varies by provider (MCP normalisers often cap around 600 characters)
- **structured_payload** — Full original JSON (stored in DB, not sent to LLM)

**Reviewer prompt:** When building the review prompt, `rag_document` sources use the **entire** stored chunk excerpt (up to the RAG chunk cap) so PDF-backed claims can be checked against the same text the researcher saw. Other providers use a bounded excerpt in the review prompt (see `app/reviewer.py`).

---

## MCP Tool Servers

Four FastAPI-based MCP servers provide the agent's data-gathering tools. Each exposes a `/tools` manifest, `/call` endpoint, and `/health` check.

### Yahoo Finance (Port 8001)

Market data via the `yfinance` library. No API key required.

| Tool | Arguments | Returns |
|---|---|---|
| `get_ticker_info` | `symbol` | Company info, price, market cap, P/E, sector, beta, 52-week range, analyst targets |
| `get_ticker_news` | `symbol`, `count=10` | Recent news articles with titles, publishers, and links |
| `get_price_history` | `symbol`, `period=1mo`, `interval=1d` | OHLCV historical price data |
| `search` | `query`, `search_type=all` | Search Yahoo Finance for quotes and news |

### FRED (Port 8002)

Federal Reserve Economic Data via the `fredapi` library. Requires `FRED_API_KEY`.

| Tool | Arguments | Returns |
|---|---|---|
| `get_series` | `series_id`, `limit=24` | Economic time series data (e.g. FEDFUNDS, UNRATE, GDP) |
| `search_series` | `query`, `limit=10` | Full-text search for FRED series by keyword |
| `get_series_info` | `series_id` | Series metadata — title, frequency, units, last updated |

### Financial Datasets (Port 8003)

Financial statement data via the [financialdatasets.ai](https://financialdatasets.ai) API. Requires `FINANCIAL_DATASETS_API_KEY`.

| Tool | Arguments | Returns |
|---|---|---|
| `get_income_statement` | `ticker`, `period=annual`, `limit=4` | Revenue, net income, EPS, margins |
| `get_balance_sheet` | `ticker`, `period=annual`, `limit=4` | Assets, liabilities, equity |
| `get_cash_flow_statement` | `ticker`, `period=annual`, `limit=4` | Operating, investing, financing cash flows |
| `get_financial_metrics` | `ticker`, `period=annual`, `limit=4` | P/E, EV/EBITDA, ROE, margins, growth rates |
| `get_financial_metrics_snapshot` | `ticker` | Latest single-point financial metrics |
| `get_stock_prices` | `ticker`, `interval=day`, `interval_multiplier=1`, `limit=30` | Historical prices |
| `get_stock_price_snapshot` | `ticker` | Latest price |
| `get_company_facts` | `ticker` | Company description, sector, employees, market cap |
| `get_insider_trades` | `ticker`, `limit=20` | Recent insider buy/sell transactions |
| `get_news` | `ticker`, `limit=10` | Company news articles |
| `get_analyst_estimates` | `ticker`, `period=annual`, `limit=4` | Revenue and EPS consensus estimates |
| `get_filings` | `ticker`, `filing_type`, `limit=10` | SEC filing metadata |
| `get_filing_items` | `accession_number` | Specific filing sections |
| `screen_stocks` | Various filters | Screen stocks by financial criteria |

### Open Web Search (Port 8004)

General web search combining multiple sources. No API key required.

| Tool | Arguments | Returns |
|---|---|---|
| `web_search` | `query`, `limit=10` | Multi-source results from Bing News RSS + DuckDuckGo |
| `fetch_web` | `url`, `max_chars=8000` | Fetch and extract text from a web page |

### Researcher Tool Whitelist

Not all MCP tools are exposed to the LLM. The researcher filters to these 9 tools to stay within context limits:

```
get_ticker_info, get_ticker_news, get_price_history,
get_series, search_series,
get_income_statement, get_financial_metrics, get_analyst_estimates,
web_search
```

---

## API Reference

### `POST /research`

Start a new research run.

**Request:**
```json
{
  "query": "What is the current state of Apple stock?",
  "output_style": "memo"
}
```

Optional fields for user-provided evidence (RAG before / alongside MCP):

| Field | Type | Effect |
|---|---|---|
| `documents` | `list[str]` | Inline document bodies; chunked and retrieved like files |
| `documents_folder` | `string` | Path to a directory of text-capable files (e.g. PDFs) indexed for retrieval |

`output_style` options: `memo` (default — executive summary + sections), `brief` (2-4 sentences), `full` (detailed report).

**Response:**
```json
{
  "run_id": "2b7b39b8b15241a8affc475e2872a63d",
  "status": "completed",
  "markdown": "# Apple Stock Analysis\n...",
  "error": null
}
```

### `GET /runs/{run_id}`

Retrieve full run details including claims, reviews, and report.

**Response:**
```json
{
  "run": { "run_id": "...", "query_text": "...", "status": "completed", "created_at": "..." },
  "claims": [ { "claim_id": "...", "claim_text": "...", "source_ids_json": "...", "support_type": "direct" } ],
  "reviews": [ { "claim_id": "...", "verdict": "verified", "notes": "...", "final_source_ids_json": "..." } ],
  "report": { "title": "...", "report_markdown": "..." }
}
```

### `GET /runs/{run_id}/sources`

Retrieve all sources collected during a run.

### `DELETE /runs/{run_id}`

Delete a run and all associated data (sources, claims, reviews, report).

---

## Project Structure

```
Finance_research_agent/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── api.py               # REST endpoints
│   ├── orchestrator.py      # Pipeline coordinator
│   ├── planner.py           # Query decomposition
│   ├── researcher.py        # Tool-calling research loop
│   ├── reviewer.py          # Claim verification (full RAG excerpts in prompt)
│   ├── formatter.py         # Structured report builder
│   ├── renderer.py          # Markdown renderer
│   ├── rag.py               # Optional document RAG phase
│   ├── rag_corpus.py        # Corpus loading from inline text / folder
│   ├── rag_vector_store.py  # Optional Chroma indexing & query
│   ├── config.py            # Settings & env vars
│   ├── db.py                # SQLite persistence
│   ├── models.py            # DB row models
│   ├── schemas.py           # Pydantic schemas
│   ├── llm_client.py        # LLM client (OpenAI-compatible)
│   ├── mcp_client.py        # MCP tool routing & execution
│   ├── source_normalizer.py # Raw response → SourceRecord
│   └── trace.py             # Debug trace logger
├── mcp_servers/
│   ├── yahoo_finance/       # Port 8001
│   ├── fred/                # Port 8002
│   ├── financial_datasets/  # Port 8003
│   └── open_websearch/      # Port 8004
├── prompts/
│   ├── planner.txt          # Planner system prompt
│   ├── researcher.txt       # Researcher system prompt
│   ├── reviewer.txt         # Reviewer system prompt
│   ├── formatter.txt        # Formatter system prompt
│   └── rag.txt              # RAG extraction system prompt
├── data/                    # SQLite database
├── reports/                 # Saved markdown reports
├── logs/                    # Debug trace & MCP server logs
├── requirements.txt
├── start_mcp_servers.sh
├── stop_mcp_servers.sh
└── .env
```
