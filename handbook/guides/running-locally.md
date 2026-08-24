# Running Encore locally

Everything runs on your machine. Supabase stays hosted: it holds Postgres, auth,
and the pgvector index, so the only thing you need from the network is a working
internet connection and the keys in `backend/.env`.

## Backend

```bash
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

Serves on `http://localhost:8000`. CORS already allows `http://localhost:5173`,
which is Vite's default port, so no extra configuration is needed to talk to the
frontend.

Console prints a `daily_quota` line after every model response showing how many
of the day's requests remain.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Serves on `http://localhost:5173`. `frontend/.env` already points
`VITE_API_BASE_URL` at `http://localhost:8000`, so the two connect with no
further wiring.

## How the two are linked

```
browser  ──  Supabase Auth  ──►  JWT
   │
   │  POST http://localhost:8000/chat/stream   (Authorization: Bearer <JWT>)
   ▼
FastAPI  ──  verifies the JWT against Supabase
   │
   ├─ agent runs: retrieval + XBRL + market + valuation tools
   ├─ grounding validator checks every citation
   └─ streams AI SDK message parts back
```

Only three variables tie them together, all already set:

| Where | Variable | Value |
|---|---|---|
| `frontend/.env` | `VITE_API_BASE_URL` | `http://localhost:8000` |
| `backend/.env` | `ALLOWED_ORIGINS` | `http://localhost:5173` |
| both | Supabase URL + anon key | same project |

To move the backend to another port, change it in both places and nothing else.

## Smoke tests, without starting either server

```bash
cd backend
.venv/Scripts/python.exe -u scripts/smoke_agent.py       # one full agent turn
.venv/Scripts/python.exe -u scripts/smoke_retrieval.py   # retrieval only, no LLM
```

`smoke_retrieval.py` costs no model quota, so reach for it first when the
question is whether search works.

## Evaluation

```bash
cd backend
.venv/Scripts/python.exe -m app.verification.evals.finance_agent_eval        # all cases
.venv/Scripts/python.exe -m app.verification.evals.finance_agent_eval 1 3    # selected cases
```

Grading is deterministic, so only the agent runs cost quota. A full pass is
roughly 40 requests against a 500/day free-tier ceiling, so prefer single cases
while iterating.
