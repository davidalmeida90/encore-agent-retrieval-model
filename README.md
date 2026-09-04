# Encore

A finance agent that answers questions about US public companies from two kinds of source, and refuses to answer from memory.

- **What management said** — retrieval over indexed SEC filings, with every claim cited back to the chunk it came from
- **What the numbers are** — SEC XBRL facts, market prices, and Treasury yields, pulled live
- **What it should be worth** — DCF and trading comparables, computed by a deterministic engine rather than by the model

Runs locally. Postgres, auth, and vector storage are hosted on Supabase; everything else runs on your machine. The model can be Gemini, Ollama, or an open model on a GPU you rent by the hour.

## Walkthroughs

Two videos, in order. The first builds the agent; the second replaces the model
behind it with an open one you host yourself, which is what `cloud/` in this
repo does.

**Part 1 — Building a Finance AI Agent App with Pydantic AI + RAG.** 33 minutes,
first principles: RAG, chunking a 10-K, hybrid retrieval and re-ranking, the
Pydantic AI agent loop, tool calls, the out-of-loop verifier, and what each part
costs.

[![Building a Finance AI Agent App with Pydantic AI + RAG](https://img.youtube.com/vi/71hV5jWe88Y/hqdefault.jpg)](https://youtu.be/71hV5jWe88Y)

**Part 2 — Open Source LLMs for Finance: Qwen 3.8, Hugging Face and RunPod.**
Picking the model on Hugging Face, quantisation and FP8, dense against mixture
of experts and why MoE disappointed on agentic work, renting an H100, weights to
disk and then into VRAM with real timings, and a Microsoft DCF run end to end on
the self-hosted model including where it got the assumptions wrong.

[![Open Source LLMs for Finance: Qwen 3.8, Hugging Face and RunPod](https://img.youtube.com/vi/yt9nrim22Gg/hqdefault.jpg)](https://youtu.be/yt9nrim22Gg)

## Design rule

The model never does arithmetic on retrieved prose. Every figure comes from a tagged XBRL fact or a valuation engine, because arithmetic-over-text is the documented FinanceBench failure mode. Everything the model *does* say about a filing must carry a citation that survives a fail-closed grounding check, or the answer is discarded rather than shown.

## Three ways to find what a filing says

The mode is picked in the UI, next to the model. It changes which tools the
model is offered and nothing else: the instructions, the output schema, the loop
guards and the grounding check are shared, so the three are comparable on the
same questions. Defined in
[backend/app/retrieval/modes.py](backend/app/retrieval/modes.py).

| Mode | How it finds text | Reaches | Costs |
|---|---|---|---|
| **RAG** (default) | Hybrid search over the ingested index | the 10 companies and years ingested | ~7-30k input tokens per question |
| **Agentic** | No index. The model searches SEC EDGAR full-text, fetches the filing, reads it | any public filer, never stale | whole documents through the model |
| **CAG** | No search. One complete 10-K sits in the context window | one company, one year, per conversation | ~90s once, then seconds |

**RAG** is the default because it is the only one whose citations point at a
chunk id the validator can check.

**Agentic retrieval** is a faithful port of the four tools in Vals AI's
`finance-agent-v2` — `web_search`, `edgar_search`, `parse_html_page`,
`retrieve_information` — so its results can be read against that benchmark. The
deviations are documented at the top of
[backend/app/agent/tools/agentic.py](backend/app/agent/tools/agentic.py).
Fetches are guarded against SSRF: the hostname is resolved, private, loopback
and link-local addresses are refused, and the request is pinned to the address
that was checked.

**CAG** ([Cache-Augmented Generation](https://arxiv.org/abs/2412.15605)) deletes
the selection step rather than improving it. The filing is injected as
*instructions*, not as a tool result, which is the whole trick: instructions
precede the question, so the 190,000 tokens form a stable prefix that vLLM's
`--enable-prefix-caching` reuses across every later turn. Measured on an H100
with the NVDA 10-K, the first question paid ~93s to prefill and the next two came
back in 45s and 33s with **99.96% of the context served from cache**. It has no
retrieval tool at all, deliberately. It is not for valuations: a DCF needs enough
rounds to reach 590,000 tokens against a 400k ceiling.

### What the hybrid stack actually does

[backend/app/retrieval/retriever.py](backend/app/retrieval/retriever.py) runs two
legs in parallel and fuses them:

```
question
  ├── HyDE expansion ──► dense leg    pgvector HNSW, cosine
  └── plainto_tsquery ─► lexical leg  Postgres GIN, BM25 rescore
                    │
                    ├── reciprocal rank fusion (k=60)
                    └── cross-encoder rerank ──► 6 passages
```

- **HyDE** ([hyde.py](backend/app/retrieval/hyde.py)) writes a hypothetical
  filing paragraph and embeds that instead of the question, because a question
  and the sentence answering it do not sit near each other in embedding space.
  It feeds only the dense leg, and fails open.
- **BM25** ([bm25.py](backend/app/retrieval/bm25.py)) replaces `ts_rank_cd`,
  which is cover density and knows nothing about how rare a term is. Supabase
  ships neither `pg_search` nor VectorChord-BM25, so IDF, k1 saturation and
  length normalisation are computed in Python from `ts_stat` document
  frequencies and the positions already stored in each `tsvector`. Nothing is
  re-tokenised; query terms are stemmed by `plainto_tsquery` so they match the
  index exactly.
- **Reranking** runs on the GPU when one is present: 6.55s to 0.2s for the same
  30 passages.

Measured with [the retrieval eval](backend/app/verification/evals/retrieval_eval.py),
recall@50 on questions whose answer chunk is known by construction:

```
                        start   OR fix   pool=50   +HyDE   +BM25
dense leg                 46%      46%       46%     58%     58%
lexical leg                2%      23%       23%     23%     42%
fused candidate pool      37%      37%       46%     50%     62%
```

The single largest fix was not a new retriever. `plainto_tsquery` joins terms
with `AND`, so a natural-language question matched **0 of 27,222 chunks**;
rewriting it to `OR` took lexical recall from 2% to 23%. The remaining problem is
now the last step rather than the first: 62% of answers reach the pool and about
30% survive the cross-encoder into the six passages the agent is shown.

## Running the model on a rented GPU

The agent talks to any OpenAI-compatible endpoint, so the model behind it can be
Gemini, Ollama on your own machine, or a GPU rented by the hour. Nothing in
`backend/` knows the difference.

```
start-encore.bat        rent the GPU, start the app, open the browser
status-encore.bat       what is running and what it has cost
stop-encore.bat         terminate everything
```

Or without the batch file:

```bash
cd path/to/repo
backend\.venv\Scripts\python.exe cloud\launch.py
```

[cloud/launch.py](cloud/launch.py) provisions the pod, reads vLLM's own log to
draw real progress bars for the 31 GB download and the shard load into VRAM,
waits for the first real completion, and then opens the browser. **Closing the
window terminates the pod**, on Ctrl+C and on the window's close button alike,
because a GPU bills whether or not anyone is watching it.

Two flags decide whether this works at all, and both are covered in
[cloud/README.md](cloud/README.md): `--tool-call-parser qwen3_xml` (with the
wrong parser vLLM hands back the tool call as ordinary text and the agent looks
like it is refusing to use its tools) and `--max-model-len`, which CAG pushes to
262,144.

Speculative decoding is deliberately **off**. Qwen3.8-27B is a hybrid
attention/GDN model, and vLLM issues #39273 and #39809 report corrupted output
with ngram speculation on that family.

Cost, measured rather than quoted:

| GPU | Rate | Decode |
|---|---|---|
| H100 NVL | $3.19/hr | 79.1 tok/s |
| RTX PRO 6000 | $2.09/hr | 46.3 tok/s |

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
│   │       ├── retrieval.py     "what did management say"   (RAG mode)
│   │       ├── agentic.py       the same question, no index   (agentic mode)
│   │       ├── cag.py           the whole filing, no search   (CAG mode)
│   │       ├── fundamentals.py  "what is the number"      (SEC XBRL)
│   │       ├── market.py        "what is it worth today"  (prices, yields)
│   │       ├── valuation.py     "what should it be worth" (DCF, comps)
│   │       ├── skills.py        methodology, loaded on demand
│   │       └── _guards.py       loop guards against runaway tool calls
│   ├── retrieval/           ← hybrid search
│   │   ├── modes.py             RAG / agentic / CAG: which tools each offers
│   │   ├── retriever.py         both legs, in parallel
│   │   ├── queries.py           the SQL: pgvector KNN and full-text
│   │   ├── bm25.py              IDF + saturation + length norm, in Python
│   │   ├── hyde.py              hypothetical-document expansion, dense leg only
│   │   ├── fusion.py            reciprocal rank fusion
│   │   └── rerank.py            cross-encoder, GPU when there is one
│   ├── verification/        is the answer right?
│   │   ├── validator.py         this answer: fail-closed citation checking
│   │   └── evals/               the agent overall: regression suite
│   │       └── retrieval_eval.py    recall per leg, ground truth by construction
│   ├── turn/                one chat turn: orchestrate, validate, stream, persist
│   ├── api/ auth/ database/ schemas/
├── vendor/                  valuation + SEC math from HKUDS/Vibe-Trading (MIT)
├── skills/                  SKILL.md methodology files
├── ingest/                  filings → chunks → embeddings
├── migrations/              Alembic schema history
├── scripts/                 ask.py, benchmark.py, smoke checks
└── tests/
cloud/                       rent a GPU and serve the model on it
├── launch.py                one command: pod, progress bars, browser, shutdown
├── serve_on_runpod.py       up / status / down, and the vLLM flags that matter
└── README.md                Ollama, RunPod, and why thinking mode is a cost
corpus/                      SEC filings: download, convert (gitignored payload)
handbook/                    architecture notes and setup guides
frontend/                    Vite + React SPA
start-encore.bat             double-click: GPU up, app up, browser open
stop-encore.bat              terminate the pod
status-encore.bat            what is running, and what it has cost
```

## Where to change things

| To change | Edit |
|---|---|
| Token ceilings, quota limits, loop guards, payload sizes | [backend/app/agent/tuning.py](backend/app/agent/tuning.py) |
| Which tools the model can call | [backend/app/agent/tools/\_\_init\_\_.py](backend/app/agent/tools/__init__.py) |
| How the agent is told to behave | [backend/app/agent/prompts/instructions.md](backend/app/agent/prompts/instructions.md) |
| Valuation methodology the model follows | [backend/skills/](backend/skills/) |
| How answers are checked before you see them | [backend/app/verification/validator.py](backend/app/verification/validator.py) |
| Which retrieval modes exist and what each may call | [backend/app/retrieval/modes.py](backend/app/retrieval/modes.py) |
| Retrieval knobs: rerank pool, HyDE on/off, BM25 on/off | [backend/app/config.py](backend/app/config.py) |
| Which GPU, which model, which vLLM flags | [cloud/serve_on_runpod.py](cloud/serve_on_runpod.py) |

## Prerequisites

| Tool | Version | Used for |
|---|---|---|
| [Python](https://www.python.org/downloads/) | 3.12+ | Backend, ingestion, corpus scripts |
| [Node.js](https://nodejs.org/) | 20+ (LTS) | Frontend toolchain |
| [pnpm](https://pnpm.io/installation) | latest | Frontend packages (`corepack enable`) |

Accounts and keys, all of which have a usable free tier:

| Service | Used for | Free tier |
|---|---|---|
| [Supabase](https://supabase.com) | Postgres, `pgvector`, auth | yes |
| [Google AI Studio](https://aistudio.google.com/apikey) | Gemini: agent, keywords, grounding judge | 500 requests per model per day |
| [OpenAI](https://platform.openai.com/api-keys) | embeddings only | no, but a full corpus embed costs cents |
| [RunPod](https://runpod.io) | optional: serve an open model yourself | no, billed by the second |

SEC EDGAR needs no key. It asks only that you identify yourself in a
`User-Agent`, which is what `SEC_EDGAR_UA` is for.

A CUDA GPU is optional. Re-ranking falls back to CPU automatically; the
difference measured here was 6.55s to 0.2s for the same 30 passages.

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

## Building the corpus

The repository ships no filings: they are 147 MB, and reproducible. Edit
`TICKERS` and `FILINGS_PER_COMPANY` at the top of `corpus/download.py`, set
`SEC_EDGAR_UA` to a real contact address, then:

```bash
python corpus/download.py             # fetch 10-K HTML from SEC EDGAR
python corpus/convert_to_markdown.py  # extract text and tables
cd backend
python -m ingest.load_source_documents
python -m ingest.chunk_and_embed --all
```

The default list is ten large caps and the last two to three annual reports of
each: 23 filings, 27,222 chunks. Embedding that costs a few cents of OpenAI
usage and no Gemini quota.

Downloaded and converted files are gitignored. Only the scripts are versioned,
so the corpus is always rebuilt rather than shipped.

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

Retrieval is measured separately, because a wrong answer from a good passage and
a wrong answer from no passage need different fixes:

```bash
.venv/Scripts/python.exe -m app.verification.evals.retrieval_eval
```

Ground truth is by construction rather than by hand: a chunk is sampled, a
question is written whose answer is only in that chunk, and recall is whether
that chunk comes back. Each question exists twice, verbatim and paraphrased,
because the gap between the two is where the dense and lexical legs disagree.
Costs no model quota beyond the HyDE expansion.

Run it more than once before believing any figure. Cost is a property of what
the model decides to do, not of the question: the same comparison question has
been measured at 2 requests and 9,888 tokens, and at 11 requests and 83,366.

## Quota

Running on Gemini's free tier means **500 requests per project per model per day**, resetting at midnight Pacific. Both budgets in `tuning.py` are set against that, and the daily count is logged after every model response, so the allowance no longer drains invisibly.

## License

MIT. See [LICENSE](LICENSE).

## Attribution

`backend/vendor/` is vendored from [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) under the MIT License. See [backend/vendor/LICENSE](backend/vendor/LICENSE). Valuation and data logic there is unmodified; only import paths and three configuration accessors were changed.
