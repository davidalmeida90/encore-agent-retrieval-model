# Backend — agent notes

FastAPI service for Encore. Read [../AGENTS.md](../AGENTS.md) first for universal
rules. This file adds backend-specific conventions.

## Read this before changing anything

These are not style preferences. Each one is a mistake that was actually made
here, cost real tokens or a wrong number, and took a while to find.

### 1. Never reimplement what `vendor/` already does

`vendor/` is 21,500 lines of valuation and SEC data code from HKUDS/Vibe-Trading.
Every serious bug in this project came from writing a "small helper" that already
existed there and behaved differently:

| What was hand-written | What already existed | Damage |
|---|---|---|
| a concept→XBRL tag map | `SEC_CONCEPT_MAP` | net debt off by $84B |
| a terminal-value bridge | `run_dcf()` in `quantlib/valuation/dcf.py` | wrong valuation |
| an annual-series builder | `load_fundamental_panel(pit=True)` | lost point-in-time safety |

**Before writing any wrapper, print the target's field NAMES and TYPES and its
required-field constants.** Reading the names alone is how floats got passed
where `FlowMetricPeriods` was required.

### 2. Type every tool parameter. Docstrings do not work.

Every tool failure traced back to a free-text parameter, and no amount of
docstring prose ever fixed one. Types fixed each permanently:

```
field names guessed wrong  ->  Literal[...] enum
peers passed as JSON text  ->  list[Peer], a real pydantic model
units silently mixed       ->  magnitude guard + billions-only contract
```

Same lesson bit the retrieval layer: three sentences in `instructions.md`
forbidding UUID citation markers were ignored, because `search_filings` printed
chunk ids as `[uuid]` and the model copied the shape. Changing the format to
`chunk_id=<uuid>` fixed it. **Structure beats instruction, every time.**

### 3. Tunable numbers live in `app/agent/tuning.py`, nowhere else

Token ceilings, quota limits, loop guards, payload thresholds, compaction, and
validation attempts. If you are about to hardcode a number that someone might
want to change while testing, it belongs there with the observation that set it.

Retrieval knobs (`retrieval_top_k`, `retrieval_rrf_k`, `retrieval_candidate_k`,
`retrieval_neighbor_radius`) are the exception: they live in `app/config.py`
because they are env-overridable deployment config.

### 4. Add or remove tools in `app/agent/tools/__init__.py`

`TOOLS` is the registry. `agent.py` passes it straight through, so a tool exists
exactly when it is listed. Comment a line to disable one during testing.

Registered tools cost schema tokens on **every** request whether called or not
(twelve tools ≈ 2,550 tokens). A tool earns its place by being called often
enough to justify that rent.

### 5. Tool returns are re-sent on every later round

A tool return persists in message history, so an oversized one is paid for
repeatedly rather than once. `search_filings` returns a measured 12,030
characters. `ToolOutputLimits` bands count **characters** unless built with
`over_tokens=True`; always keep the clamp BELOW its own trigger, or the band
does nothing.

### 6. Free tier is 500 requests per day

`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, resetting midnight Pacific.
A full eval pass costs about 40. Prefer `scripts/smoke_retrieval.py` (no LLM) and
single eval cases while iterating.

## Stack

- Python 3.12+, FastAPI + uvicorn, Pydantic v2 + pydantic-settings
- **PydanticAI** owns the agent loop; `pydantic-ai-harness` supplies capabilities
  (tool output limits, compaction, spend budgets)
- **Gemini** for chat, grounding judge, and keyword extraction
- **OpenAI** for embeddings only (`text-embedding-3-small`, 1536d)
- Supabase Postgres + `pgvector`; SQLAlchemy models + Alembic migrations
- Hybrid retrieval: vector and full-text queried separately, fused in Python with
  Reciprocal Rank Fusion
- `structlog` for logging, `pytest` for tests

Dependencies are declared in `pyproject.toml`. Note `uv` is **not** installed on
this machine; call the interpreter directly at `.venv/Scripts/python.exe`.

## Layout

```text
backend/
├── app/
│   ├── main.py          # FastAPI entrypoint
│   ├── config.py        # pydantic-settings, single source of truth for env
│   ├── telemetry.py     # per-turn call accounting + daily quota counter
│   ├── api/             # routers (chat, auth) — thin
│   ├── auth/            # Supabase JWT verification
│   ├── chat/            # turn orchestration, streaming, AI SDK message conversion
│   ├── agent/           # the agent
│   │   ├── agent.py     #   model + instructions + capabilities + tools
│   │   ├── tuning.py    #   every knob worth turning
│   │   ├── deps.py      #   per-turn context (retriever, citation registry)
│   │   ├── outputs.py   #   GroundedAnswer schema
│   │   ├── prompts/     #   system instructions
│   │   └── tools/       #   TOOLS registry + one module per domain
│   ├── retrieval/       # pgvector + full-text, RRF fusion, passage hydration
│   ├── grounding/       # fail-closed citation validation
│   ├── database/        # SQLAlchemy models, Supabase client, query helpers
│   └── schemas/         # API request/response DTOs
├── vendor/              # HKUDS/Vibe-Trading (MIT) — do not edit, see LICENSE
├── skills/              # SKILL.md methodology, loaded on demand
├── ingest/              # PDF -> chunks -> embeddings
├── evals/               # regression suite (deterministic grading)
├── scripts/             # smoke_agent.py, smoke_retrieval.py
├── tests/
└── pyproject.toml
```

## Code style (backend-specific)

- **Type hints on public functions and module-level things.** Don't annotate every local.
- **Async by default in request-path code.** Don't run blocking I/O on the event loop. Network calls must be async.
- **Use `async def` for all route handlers** and any I/O service function.
- **Validate at boundaries only.** HTTP input is validated by Pydantic models. External API responses are validated when parsed. Internal callers are trusted.

## Configuration

- `app.config.settings` is the single source of truth for environment. Import
  settings where needed; never call `os.getenv` in app code, never call `load_dotenv`.
- Never print the contents of `.env`. Keys have been leaked into a transcript
  this way before.
