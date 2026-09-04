"""BM25 over the Postgres full-text index, without a Postgres extension.

## Why this exists
The lexical leg scores with `ts_rank_cd`, which is cover density: it rewards
term frequency and proximity, and knows nothing about how rare a term is. So in
"Azure competitive advantage" the word *advantage* counts about as much as
*Azure*, and in a corpus where JPMorgan is 34% of all chunks, banking vocabulary
is treated as informative in an Apple question.

BM25 fixes that with three terms `ts_rank_cd` lacks:

    IDF          a rare word dominates a common one
    saturation   the twentieth "revenue" adds almost nothing (k1)
    length norm  a long chunk does not win by containing more words (b)

The usual way to get it in Postgres is ParadeDB's `pg_search` or
VectorChord-BM25, neither of which is in Supabase's extension set. So it is
computed here instead, over the candidates Postgres already returned.

## How, without reimplementing text search
Everything needed is already in the index and none of it requires re-tokenising:

    df, per lexeme    ts_stat over search_vector, 10,761 terms in ~3s, cached
    tf, per document  count of positions in the stored tsvector
    document length   total positions in that tsvector
    query lexemes     plainto_tsquery, so stemming matches the index exactly

Re-stemming in Python would be the obvious shortcut and would quietly disagree
with the index on every irregular word.

## What this is for
An experiment first. `ts_rank_cd` and BM25 can be scored over the same candidate
set and compared on the retrieval eval, so the choice is a measurement rather
than an argument. Swap it in only if it wins.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings

# Robertson's defaults. k1 controls how fast term frequency saturates, b how
# hard length normalisation bites. Worth tuning only after BM25 has been shown
# to beat what is already there.
K1 = 1.2
B = 0.75

_LEXEME = re.compile(r"'((?:[^']|'')+)':([\d,]+)")


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Everything BM25 needs about the corpus, computed once."""

    n_docs: int
    doc_freq: dict[str, int]
    avg_len: float

    def idf(self, term: str) -> float:
        """Robertson-Sparck Jones IDF, the smoothed form used by Lucene.

        A term in every document scores ~0; a term in two documents scores high.
        `+1` inside the log keeps it non-negative for terms in over half the
        corpus, which the unsmoothed form makes negative.
        """
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))


@lru_cache(maxsize=1)
def _stats_cached(marker: int) -> CorpusStats:
    from app.database.session import get_session

    with get_session() as session:
        n_docs = session.execute(text("SELECT count(*) FROM document_chunks")).scalar() or 0
        rows = session.execute(text(
            "SELECT word, ndoc, nentry FROM ts_stat("
            "'SELECT search_vector FROM document_chunks')"
        )).fetchall()
    doc_freq = {word: ndoc for word, ndoc, _ in rows}
    total_terms = sum(nentry for _, _, nentry in rows)
    return CorpusStats(
        n_docs=n_docs,
        doc_freq=doc_freq,
        avg_len=(total_terms / n_docs) if n_docs else 1.0,
    )


def corpus_stats() -> CorpusStats:
    """Cached corpus statistics. The marker exists so the cache can be busted."""
    return _stats_cached(0)


def _parse_vector(vector: str) -> tuple[dict[str, int], int]:
    """Term frequencies and length from a stored tsvector's text form.

    A tsvector renders as ``'lexeme':1,7,19 'other':3``, so the count of
    positions is the term frequency and their sum is the document length.
    """
    frequencies: dict[str, int] = {}
    length = 0
    for lexeme, positions in _LEXEME.findall(vector or ""):
        count = positions.count(",") + 1
        frequencies[lexeme.replace("''", "'")] = count
        length += count
    return frequencies, length


def query_lexemes(session: Session, question: str) -> list[str]:
    """Stem the question through Postgres, so it matches the index exactly."""
    rendered = session.execute(
        text("SELECT plainto_tsquery(:cfg, :q)::text"),
        {"cfg": settings.retrieval_fts_config, "q": question},
    ).scalar() or ""
    return [m.group(1).replace("''", "'")
            for m in re.finditer(r"'((?:[^']|'')+)'", rendered)]


def score_rows(
    session: Session,
    question: str,
    rows: list[tuple],
    *,
    top_k: int,
) -> list:
    """BM25 over rows already fetched as (id, tsvector-text), best first.

    The preferred entry point. `rescore` below takes ids instead and pays a
    second round trip for the vectors, which was 83% of its cost.
    """
    if not rows:
        return []
    stats = corpus_stats()
    terms = query_lexemes(session, question)
    if not terms:
        return [row[0] for row in rows][:top_k]

    scored: list[tuple[float, object]] = []
    for chunk_id, vector in rows:
        frequencies, length = _parse_vector(vector)
        if not length:
            continue
        norm = K1 * (1.0 - B + B * length / stats.avg_len)
        total = 0.0
        for term in terms:
            tf = frequencies.get(term, 0)
            if tf:
                total += stats.idf(term) * (tf * (K1 + 1.0)) / (tf + norm)
        scored.append((total, chunk_id))
    scored.sort(key=lambda pair: -pair[0])
    return [chunk_id for _score, chunk_id in scored[:top_k]]


def rescore(
    session: Session,
    question: str,
    chunk_ids: list,
    *,
    top_k: int,
) -> list:
    """Reorder candidate chunk ids by BM25, best first.

    Takes ids rather than running its own search: the candidate set should come
    from the same index either way, so that a comparison against `ts_rank_cd`
    differs only in the scoring.
    """
    if not chunk_ids:
        return []
    stats = corpus_stats()
    terms = query_lexemes(session, question)
    if not terms:
        return list(chunk_ids)[:top_k]

    rows = session.execute(
        text("SELECT id, search_vector::text FROM document_chunks WHERE id = ANY(:ids)"),
        {"ids": list(chunk_ids)},
    ).fetchall()

    scored: list[tuple[float, object]] = []
    for chunk_id, vector in rows:
        frequencies, length = _parse_vector(vector)
        if not length:
            continue
        norm = K1 * (1.0 - B + B * length / stats.avg_len)
        score = 0.0
        for term in terms:
            tf = frequencies.get(term, 0)
            if tf:
                score += stats.idf(term) * (tf * (K1 + 1.0)) / (tf + norm)
        scored.append((score, chunk_id))

    scored.sort(key=lambda pair: -pair[0])
    return [chunk_id for _score, chunk_id in scored[:top_k]]
