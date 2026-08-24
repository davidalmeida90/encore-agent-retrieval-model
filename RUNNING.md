# Running Encore

Everything runs on this machine. Supabase stays hosted (Postgres, pgvector, auth),
so you need an internet connection and the keys in `backend/.env`, but no cloud
deploy of your own.

## Start

```powershell
cd path\to\encore
.\start.ps1
```

That single command:

1. frees ports 8000 and 5173 if something is still holding them
2. starts the backend on `http://localhost:8000`
3. starts the frontend on `http://localhost:5173`
4. waits for the backend to answer `/health` (about 15s, it loads the reranker)
5. opens the browser

Options:

```powershell
.\start.ps1 -NoBrowser                       # do not open a tab
.\start.ps1 -BackendPort 8001 -FrontendPort 5174
```

If PowerShell refuses to run it:

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

## Stop

**Press `Ctrl+C` in that window**, or just close the window. Both servers are
children of it and are killed on the way out, including the vite process that
`npm` spawns as a grandchild.

Closing the *browser tab* does not stop anything. A browser cannot reliably tell
a server it is going away: `beforeunload` also fires on an ordinary refresh, and
never fires at all if the browser crashes or the tab is discarded. The terminal
window is the honest control point.

If a port is somehow still held after a crash:

```powershell
Get-NetTCPConnection -LocalPort 8000,5173 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## Log in

Sign up at `http://localhost:5173/signup` with any email and password. Accounts
live in your own Supabase project's `auth.users`, so the first one you create is
yours; there is no seeded account and no shared password.

## Watching what the agent does

The **Log** button, top right of the chat header, opens the activity panel:

```
 0.0s  analyzing  · Analyzing your question…
 1.3s  searching  · Searching SEC filings… (ticker=MSFT, fiscal_years=2024,2025)
 1.3s  tool       · search_filings(ticker=MSFT, fiscal_years=2024,2025)
 9.4s  usage      · 2 model requests · 1 tool calls · 8,011 in / 773 out tokens
 9.5s  verifying  · Verifying citations…
```

Every tool call with its arguments, elapsed time, retries, and what the run cost.
It resets per question.

## Without the browser

One question, straight from the terminal, showing tools and tokens:

```powershell
cd backend
.\.venv\Scripts\python.exe -u scripts\ask.py "What was Apple's capital expenditure in fiscal 2025?"
```

Retrieval only, no model calls and therefore no quota:

```powershell
.\.venv\Scripts\python.exe -u scripts\smoke_retrieval.py
```

Regression suite (each case spends quota, so prefer single cases while iterating):

```powershell
.\.venv\Scripts\python.exe -m app.verification.evals.finance_agent_eval 1 3
```

Tests, which cost nothing:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q -m "not integration"
```

## Adding filings to the corpus

```powershell
.\.venv\Scripts\python.exe corpus\download.py            # fetch from SEC EDGAR
.\.venv\Scripts\python.exe corpus\convert_to_markdown.py # extract text and tables
cd backend
.\.venv\Scripts\python.exe -m ingest.load_source_documents
.\.venv\Scripts\python.exe -m ingest.chunk_and_embed --all --force
```

Edit `TICKERS` and `FILINGS_PER_COMPANY` at the top of `corpus/download.py` to
change what is fetched. Embedding costs a few cents of OpenAI usage and no Gemini
quota.
