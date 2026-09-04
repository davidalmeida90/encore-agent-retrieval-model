"""## Does the lexical leg earn its place? An answer with numbers.

    python -m app.verification.evals.retrieval_eval --build 40   # author questions once
    python -m app.verification.evals.retrieval_eval             # score, no LLM needed

## The question this settles
Retrieval here is hybrid: pgvector for meaning, Postgres full-text for tokens,
fused by reciprocal rank, then reordered by a cross-encoder. The lexical half
scores with `ts_rank_cd`, which has no IDF, so a rare discriminative token like
"Azure" is weighted no more heavily than "advantage". BM25 exists to fix exactly
that, and swapping it in is real work.

Before doing that work it is worth knowing whether the lexical leg contributes
anything the dense leg does not already find. That is measurable, and this
measures it.

## How ground truth is obtained without labelling anything by hand
A chunk is picked, then a question is written FROM it. The chunk it came from is
by construction the right answer, so recall needs no judge and no opinion: either
that chunk id comes back or it does not.

Two questions are written per chunk, and the pair is the point:

    verbatim     may reuse the filing's own distinctive terms
    paraphrased  must avoid them, and say the same thing in an analyst's words

A single-style eval would flatter one leg and prove nothing. Verbatim questions
are what lexical search is for; paraphrased questions are what embeddings are
for. Reporting them separately shows where each leg actually earns its keep, and
whether either is dead weight.

## What is reported
Recall at each stage of the pipeline, per leg and fused, because the stages have
different jobs:

    semantic@50 / fts@50   candidate recall: did this leg find it at all
    fused@30               what the cross-encoder is allowed to consider
    final@6                what the model actually sees

`fused@30` is the number that decides whether BM25 is worth it. A chunk missing
there is invisible no matter how good the reranker is.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.config import settings
from app.database.session import get_session
from app.retrieval.embeddings import embed_query
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.bm25 import rescore as bm25_rescore
from app.retrieval.hyde import expand as hyde_expand
from app.retrieval.queries import full_text_search, semantic_search

HERE = Path(__file__).parent
QUESTION_FILE = HERE / "retrieval_eval_questions.json"

# Chunks that are not prose: contents pages, page furniture, exhibit indexes.
# They are lexical bait, dense with query terms and carrying no claim, and a
# question written from one would measure nothing worth knowing.
_JUNK = (
    "table of contents", "index to", "exhibit index", "signatures",
    "page intentionally", "form 10-k", "united states securities",
)


@dataclass
class Case:
    chunk_id: str
    ticker: str
    style: str  # "verbatim" or "paraphrased"
    question: str


def _sample_chunks(n: int, seed: int = 7) -> list[dict[str, Any]]:
    """Prose chunks, spread across companies rather than drawn from the biggest.

    JPM alone is a third of the corpus, so uniform sampling would measure JPM.
    """
    with get_session() as session:
        rows = session.execute(text("""
            SELECT dc.id::text AS id, dc.text, sd.ticker, sd.fiscal_year
            FROM document_chunks dc
            JOIN source_documents sd ON sd.id = dc.document_id
            WHERE length(dc.text) BETWEEN 600 AND 2500
        """)).mappings().all()

    usable = [
        dict(r) for r in rows
        if not any(marker in r["text"][:200].lower() for marker in _JUNK)
    ]
    by_ticker: dict[str, list[dict]] = {}
    for row in usable:
        by_ticker.setdefault(row["ticker"], []).append(row)

    rng = random.Random(seed)
    for pool in by_ticker.values():
        rng.shuffle(pool)

    picked: list[dict] = []
    tickers = sorted(by_ticker)
    while len(picked) < n and any(by_ticker[t] for t in tickers):
        for ticker in tickers:
            if by_ticker[ticker] and len(picked) < n:
                picked.append(by_ticker[ticker].pop())
    return picked


def build(n: int) -> None:
    """Write questions from sampled chunks, using Gemini. Run once."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.gemini_api_key,
                    base_url=settings.gemini_openai_base_url)
    chunks = _sample_chunks(n)
    cases: list[dict] = []
    for index, chunk in enumerate(chunks, 1):
        prompt = (
            "Below is one passage from an SEC 10-K filing. Write TWO questions an "
            "equity analyst would ask, each answerable from THIS passage alone.\n\n"
            "1. verbatim: you may use the filing's own distinctive wording.\n"
            "2. paraphrased: say the same thing WITHOUT reusing any distinctive "
            "noun or phrase from the passage. Use an analyst's ordinary words.\n\n"
            "Name the company in both. Return strict JSON: "
            '{"verbatim": "...", "paraphrased": "..."}\n\n'
            f"Company: {chunk['ticker']} FY{chunk['fiscal_year']}\n"
            f"---\n{chunk['text'][:2200]}\n---"
        )
        try:
            completion = client.chat.completions.create(
                model=settings.gemini_keyword_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            pair = json.loads(completion.choices[0].message.content or "{}")
        except Exception as exc:
            print(f"  [{index}/{len(chunks)}] skipped: {type(exc).__name__}")
            continue
        for style in ("verbatim", "paraphrased"):
            if pair.get(style):
                cases.append({"chunk_id": chunk["id"], "ticker": chunk["ticker"],
                              "style": style, "question": pair[style]})
        print(f"  [{index}/{len(chunks)}] {chunk['ticker']}", flush=True)

    QUESTION_FILE.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"\nwrote {len(cases)} questions from {len(chunks)} chunks -> {QUESTION_FILE.name}")


def _stage_ranks(case: Case) -> dict[str, int | None]:
    """Where the true chunk lands at each stage. None means absent."""
    # Mirror production: the dense leg embeds the HyDE expansion, the lexical
    # leg matches the user's literal words. Embedding the bare question here
    # would measure a pipeline that no longer exists.
    query_vec = embed_query(hyde_expand(case.question))
    with get_session() as session:
        dense = semantic_search(session, query_vec, limit=settings.retrieval_candidate_k)
        if settings.retrieval_bm25_enabled:
            wide = full_text_search(session, case.question,
                                    limit=settings.retrieval_bm25_candidates)
            order = bm25_rescore(session, case.question,
                                 [h.chunk_id for h in wide],
                                 top_k=settings.retrieval_candidate_k)
            by_id = {h.chunk_id: h for h in wide}
            sparse = [by_id[c] for c in order if c in by_id]
        else:
            sparse = full_text_search(session, case.question,
                                      limit=settings.retrieval_candidate_k)

    def rank_in(hits) -> int | None:
        # Position in the list, not hit.rank: BM25 reorders and the stored rank
        # is the one ts_rank_cd assigned before rescoring.
        for position, hit in enumerate(hits, 1):
            if str(hit.chunk_id) == case.chunk_id:
                return position
        return None

    # RRF takes lists of chunk ids and returns (id, score) pairs, best first.
    fused = reciprocal_rank_fusion([[h.chunk_id for h in dense],
                                    [h.chunk_id for h in sparse]])

    def rank_in_fused() -> int | None:
        for position, (chunk_id, _score) in enumerate(fused, 1):
            if str(chunk_id) == case.chunk_id:
                return position
        return None

    return {
        "semantic": rank_in(dense),
        "fts": rank_in(sparse),
        "fused": rank_in_fused(),
    }


def score() -> None:
    if not QUESTION_FILE.exists():
        raise SystemExit("No questions yet. Run with --build 40 first.")
    cases = [Case(**c) for c in json.loads(QUESTION_FILE.read_text(encoding="utf-8"))]
    pool = settings.retrieval_rerank_pool

    results: list[dict] = []
    for index, case in enumerate(cases, 1):
        try:
            ranks = _stage_ranks(case)
        except Exception as exc:
            print(f"  [{index}/{len(cases)}] error {type(exc).__name__}: {exc}"[:110])
            continue
        results.append({"style": case.style, **ranks})
        if index % 10 == 0:
            print(f"  {index}/{len(cases)}", flush=True)

    def pct(rows, key, cutoff) -> str:
        if not rows:
            return "-"
        hit = sum(1 for r in rows if r[key] is not None and r[key] <= cutoff)
        return f"{hit / len(rows):5.0%}"

    print("\n%-14s %6s %12s %12s %12s %12s" % (
        "question style", "n", f"sem@{settings.retrieval_candidate_k}",
        f"fts@{settings.retrieval_candidate_k}", f"fused@{pool}",
        f"fused@{settings.retrieval_top_k}"))
    print("-" * 74)
    for style in ("verbatim", "paraphrased", "ALL"):
        rows = results if style == "ALL" else [r for r in results if r["style"] == style]
        print("%-14s %6d %12s %12s %12s %12s" % (
            style, len(rows),
            pct(rows, "semantic", settings.retrieval_candidate_k),
            pct(rows, "fts", settings.retrieval_candidate_k),
            pct(rows, "fused", pool),
            pct(rows, "fused", settings.retrieval_top_k)))

    # The decision this eval exists to inform.
    only_fts = [r for r in results if r["fts"] is not None and r["semantic"] is None]
    only_sem = [r for r in results if r["semantic"] is not None and r["fts"] is None]
    neither = [r for r in results if r["semantic"] is None and r["fts"] is None]
    print(f"\nfound by FTS alone      {len(only_fts):3d}  <- what the lexical leg is worth")
    print(f"found by semantic alone {len(only_sem):3d}")
    print(f"found by neither        {len(neither):3d}  <- unreachable, a corpus problem")
    if results:
        print(f"\nIf 'FTS alone' is near zero the lexical leg adds nothing and BM25 "
              f"would improve\nnothing. If it is material, better lexical scoring "
              f"has something to work with.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval recall by stage.")
    parser.add_argument("--build", type=int, metavar="N",
                        help="author questions from N sampled chunks, then exit")
    args = parser.parse_args()
    if args.build:
        build(args.build)
    else:
        score()


if __name__ == "__main__":
    main()
