# Encore

A finance agent that answers questions about US public companies from two kinds of source, and refuses to answer from memory.

- **What management said** — retrieval over indexed SEC filings, with every claim cited back to the chunk it came from
- **What the numbers are** — SEC XBRL facts, market prices, and Treasury yields, pulled live
- **What it should be worth** — DCF and trading comparables, computed by a deterministic engine rather than by the model

Runs locally. Postgres, auth, and vector storage are hosted on Supabase; everything else runs on your machine.

## Design rule

The model never does arithmetic on retrieved prose. Every figure comes from a tagged XBRL fact or a valuation engine, because arithmetic-over-text is the documented FinanceBench failure mode. Everything the model *does* say about a filing must carry a citation that survives a fail-closed grounding check, or the answer is discarded rather than shown.

## Layout

```
backend/
├── app/
│   ├── main.py              FastAPI entrypoint
│   ├── config.py            settings, read from .env
│   ├── telemetry.py         per-turn call accounting + daily quota counter
│   ├── agent/               ← the agent
│   │   ├── agent.py         definition: model, instructions, capabilities, tools
│   │   ├── tuning.py        every knob worth turning during testing
│   │   ├── deps.py          per-turn context (retriever, citation registry)
│   │   ├── outputs.py       GroundedAnswer schema
│   │   ├── status.py        UI progress events
│   │   ├── prompts/         system instructions
│   │   └── tools/           ← add or disable tools in tools/__init__.py
│   │       ├── __init__.py      the TOOLS registry
│   │       ├── retrieval.py     "what did management say"
│   │       ├── fundamentals.py  "what is the number"      (SEC XBRL)
│   │       ├── market.py        "what is it worth today"  (prices, yields)
│   │       ├── valuation.py     "what should it be worth" (DCF, comps)
│   │       ├── skills.py        methodology, loaded on demand
│   │       └── _guards.py       loop guards against runaway tool calls
│   ├── retrieval/           hybrid search: pgvector + full-text, RRF, reranking
│   ├── verification/        is the answer right?
│   │   ├── validator.py         this answer: fail-closed citation checking
│   │   └── evals/               the agent overall: regression suite
│   ├── turn/                one chat turn: orchestrate, validate, stream, persist
│   ├── api/ auth/ database/ schemas/
├── vendor/                  valuation + SEC math from HKUDS/Vibe-Trading (MIT)
├── skills/                  SKILL.md methodology files
├── ingest/                  filings → chunks → embeddings
├── migrations/              Alembic schema history
├── scripts/                 ask.py, benchmark.py, smoke checks
└── tests/
corpus/                      SEC filings: download, convert (gitignored payload)
handbook/                    architecture notes and setup guides
frontend/                    Vite + React SPA
```

## Where to change things

| To change | Edit |
|---|---|
| Token ceilings, quota limits, loop guards, payload sizes | [backend/app/agent/tuning.py](backend/app/agent/tuning.py) |
| Which tools the model can call | [backend/app/agent/tools/\_\_init\_\_.py](backend/app/agent/tools/__init__.py) |
| How the agent is told to behave | [backend/app/agent/prompts/instructions.md](backend/app/agent/prompts/instructions.md) |
| Valuation methodology the model follows | [backend/skills/](backend/skills/) |
| How answers are checked before you see them | [backend/app/verification/validator.py](backend/app/verification/validator.py) |

## Running it

```bash
# backend
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

# frontend
cd frontend
npm install && npm run dev
```

Configuration lives in `backend/.env`. See [handbook/guides/backend-setup.md](handbook/guides/backend-setup.md) and [handbook/guides/supabase-setup.md](handbook/guides/supabase-setup.md).

## Evaluation

```bash
cd backend
.venv/Scripts/python.exe -m app.verification.evals.finance_agent_eval      # all cases
.venv/Scripts/python.exe -m app.verification.evals.finance_agent_eval 1 3  # selected
```

Grading is deterministic string and structure checking, so scoring costs no model quota; only the agent runs spend it. A full pass costs roughly 40 requests, so run individual cases while iterating.

To measure what a question *costs* rather than whether it is right:

```bash
.venv/Scripts/python.exe -u scripts/benchmark.py        # four questions, one per capability
.venv/Scripts/python.exe -u scripts/benchmark.py 2 4    # selected
```

Run it more than once before believing any figure. Cost is a property of what
the model decides to do, not of the question: the same comparison question has
been measured at 2 requests and 9,888 tokens, and at 11 requests and 83,366.

## Quota

Running on Gemini's free tier means **500 requests per project per model per day**, resetting at midnight Pacific. Both budgets in `tuning.py` are set against that, and the daily count is logged after every model response, so the allowance no longer drains invisibly.

## License

MIT. See [LICENSE](LICENSE).

## Attribution

`backend/vendor/` is vendored from [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) under the MIT License. See [backend/vendor/LICENSE](backend/vendor/LICENSE). Valuation and data logic there is unmodified; only import paths and three configuration accessors were changed.
