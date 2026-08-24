# Video walkthrough map

Five parts, in the order you proposed. For each: the files to open, in reading
order, and a live demo moment that proves the point rather than describing it.

Every number here was measured on this machine, not estimated.

---

## Part 1 — RAG

**Open first:** `app/retrieval/retriever.py` (333 lines)

Its docstring opens with the whole pipeline as a diagram, then a real trace of
one question. Read that on camera and the rest of the part explains itself.

| File | Lines | What to say |
|---|---|---|
| `app/retrieval/retriever.py` | 333 | Orchestrates the 5 steps. Numbered comments mark each one |
| `app/retrieval/queries.py` | 150 | The actual SQL. Show `embedding <=> vector` and `ts_rank_cd` side by side |
| `app/retrieval/fusion.py` | **18** | RRF is 18 lines. Worth showing precisely because it is so small |
| `app/retrieval/rerank.py` | 145 | The cross-encoder, and why fusion alone cannot detect absence |
| `app/retrieval/keywords.py` | 280 | Turns a question into search terms, stripping the scaffolding |

**Demo:** ask *"How does Microsoft describe Azure's competitive advantage?"* with
the Log panel open. One search, 5 citations, ~8,000 tokens.

**The two numbers that carry the part:**
- semantic returned 50 hits, full-text returned 7, and only **7 of each top-20
  overlapped**. That 13-of-20 disagreement is the whole argument for hybrid.
- fused RRF scores were 0.0318, 0.0312, 0.0306, all within 2% of each other. Rank
  fusion cannot tell a bullseye from the nearest thing available, which is
  exactly what the reranker is for.

**Optional depth — ingestion.** Its own segment if you want one:
`ingest/chunking.py` (503) and `ingest/sec_tables.py` (572). Good story: EDGAR
splits words across styled spans, so the corpus contained `B USINESS`,
`RIS K FACTORS` and `acqui ition` until it was fixed at the source. 40% of chunks
had no section; now 31 of 27,222.

---

## Part 2 — the agent harness

**Open first:** `app/agent/agent.py` (253 lines)

Docstring opens with the full turn as a diagram: question, model, tools, results
appended back into history, output validation, then the grounding gate. Then
three real runs with their costs.

| File | Lines | What to say |
|---|---|---|
| `app/agent/agent.py` | 253 | The whole agent is one `Agent(...)` call. PydanticAI owns the loop |
| `app/agent/tools/__init__.py` | 92 | The registry. A tool exists exactly when it is in this list |
| `app/agent/tools/fundamentals.py` | 267 | XBRL facts. Every parameter typed, no free text |
| `app/agent/tools/valuation.py` | 571 | 23 typed parameters, then hands off to the vendored engine |
| `app/agent/tools/_guards.py` | 92 | Loop guards. Both exist because a real run wasted real money |
| `app/agent/toolgate.py` | 61 | Hides valuation tools unless the question needs them |
| `app/agent/tuning.py` | 133 | Every knob, each carrying the observation that set it |

**Demo:** open the **Tools** panel, expand `run_dcf_valuation`. It shows the live
Python read from the file, so what you see is what is running.

**The through-line for this part:** every failure came from a free-text parameter,
and prose never fixed one of them. Types did.

```
field names guessed wrong  ->  Literal enum
peers passed as JSON text  ->  list[Peer]
units silently mixed       ->  magnitude guard + billions-only contract
```

**Best single demo:** `toolgate.py`. Two valuation tools are 1,337 of a
2,707-token schema budget, re-sent on every round. Hiding them on non-valuation
questions took a simple question from 18,435 tokens to 10,865.

---

## Part 3 — validator and evals

**Open first:** `app/verification/validator.py` (315 lines)

| File | Lines | What to say |
|---|---|---|
| `app/verification/validator.py` | 315 | Two layers: free structural checks, then a paid LLM judge |
| `app/turn/orchestrator.py` | 161 | Where the gate sits, and what a failure costs |
| `app/agent/outputs.py` | **29** | The whole contract is 29 lines |
| `app/verification/evals/finance_agent_eval.py` | 205 | 8 cases, deterministic grading, so scoring costs no quota |

**The point of the part:** the agent has no authority to publish. The validator
takes two objects and pairs them: what the model *claims* (its excerpt) against
what was *actually retrieved* (the registry). A cited chunk that was never
retrieved fails immediately, without asking the judge.

**Demo:** ask anything with the Log panel open and show the verdict line:

```
grounding · Grounding passed · 3 citations verified
grounding · Grounding passed · answer from tool data, nothing to cite
error     · Grounding FAILED (attempt 1/2): Citation [2] is not supported by...
```

**The honest moment worth including:** the log caught a real bug on its first
real use. Combined markers like `[1, 2]` did not match the validator's
`\[(\d+)\]` regex, so an answer displayed five references while one was verified.

Worth telling as two bugs rather than one, because that is what it was. Teaching
the validator to read `[1, 2]` fixed the checking; the *renderer* had the same
blind spot, so single markers became superscript links and combined ones stayed
as literal brackets in the same paragraph. One misreading, two places, and only
the second was visible on screen.

---

## Part 4 — backend and frontend basics

| File | Lines | What to say |
|---|---|---|
| `app/api/chat.py` | 192 | Thin routes. All the work is elsewhere |
| `app/auth/dependencies.py` | **69** | Supabase JWT verification, entire auth layer |
| `app/database/supabase.py` | 55 | The client wrapper |
| `app/turn/orchestrator.py` | 161 | Turn lifecycle and SSE streaming |
| `app/agent/usage.py` | 205 | Token attribution behind the Tokens panel |

**Supabase holds:** chunk text, 1536-dim embeddings, HNSW and GIN indexes, chat
threads, messages, citations, users. Show it live:

```sql
select count(*), vector_dims(embedding) from document_chunks where embedding is not null;
-- 27222, 1536
```

**Worth being precise on camera about what runs where:** vector KNN and full-text
run *inside Postgres*; RRF and the cross-encoder run locally. The reranker never
touches the database, it only re-orders what Supabase already returned.

---

## Part 5 — the frontend

Four tabs of one inspector, each answering a different question:

| Panel | Question it answers |
|---|---|
| **Activity** | What did it actually do, in what order, and did grounding pass |
| **Tools** | What can it call, and what does that code look like |
| **Tokens** | What did that cost, and what was the money spent on |
| **Corpus** | What is actually indexed, and could it have found this at all |

One panel, four tabs, behind the **Inspect** button. They were four separate
sliding panels until opening two at once squeezed the conversation to nothing.

**Best demo sequence, one question, all three panels:**

Ask *"What was Apple's capital expenditure in fiscal 2025?"*

- **Activity** shows 2 requests, 1 tool call, no search at all. A number question
  never touches retrieval.
- **Tokens** shows the split, and the system prompt row expands to the literal
  prompt being paid for on every round.
- **Tools** shows `get_sec_financials` and its real source.

Then ask the Azure question and show the contrast: one search, reranking, five
citations, grounding verified.

**A good closing point:** cost grows with the *square* of the round count,
because every round re-sends all previous rounds.

```
 3 requests ->  18,435 tokens ->  6,145 per request
10 requests -> 174,269 tokens -> 17,427 per request
```

That single fact explains nearly every optimisation in the project.

---

## Numbers you can quote, all measured here

```
corpus            27,222 chunks, 23 filings, 10 companies
search            5.71s CPU / 3.35s GPU
                  embed 0.9s | keywords 1.0s | pgvector 0.8s | FTS 0.06s
                  hydrate 2.6s | rerank 6.55s -> 1.6s (8 threads) -> 0.2s (GPU)
price history     2.36s live -> 0.11s cached
tests             91 passing
```

## Per-question cost, measured over repeat runs

Quote the RANGE, never a single figure. Same four questions, same corpus, run
several times through `scripts/benchmark.py`:

```
                    requests      input tokens        runs
Apple capex          2 - 5       7,030 -  25,547       3
Azure advantage      2 - 3       8,012 -  13,432       3
Apple vs MSFT R&D    2 - 11      9,888 -  83,366       4
Apple DCF            6 - 7      49,512 -  53,973       2
```

Two things in that table are worth saying out loud.

The "simple" question is not reliably cheap. Twice it answered from XBRL in two
requests; once it decided to search the filings as well and cost 3.6x more. Same
words, same corpus.

The DCF is the most STABLE of the four, and it is the most expensive. Its work is
done by a deterministic tool, so the model has little left to improvise. Cost
tracks how much freedom the model has, not how hard the question sounds.

The one failure in those runs is the cheapest run in the table: 2 requests, 9,888
tokens, and a `[4]` marker with no citation 4 behind it. The fast answer was the
wrong one, and the structural check caught it for free. The app retries once, so
a user would have seen a slower answer rather than that one.

## The best story in the project

The hard comparison question went from 174,000 tokens and failing to answering
correctly for a fraction of that. Most of the gain was **not** clever
engineering. Microsoft's FY2026 10-K had
never been downloaded, because this line excluded the current year:

```python
range(datetime.now(UTC).year - FILINGS_PER_COMPANY, datetime.now(UTC).year)
#                                                   ^ range() is END-EXCLUSIVE
```

XBRL knew about FY2026, the corpus stopped at FY2025, and the model burned six
searches trying to reconcile two sources that genuinely disagreed. Guards and
budgets treated the symptom. One character fixed the cause.
