"""## Hybrid retrieval: embed -> search twice -> fuse -> hydrate -> rerank

                            question
                               |
                +--------------+--------------+     1. run together (threads)
                v                             v
             embed                        keywords
          1536 floats            "datacenter capacity constraints"
                |                             |
                v                             v
          semantic (pgvector)        full-text (Postgres FTS)   2. run together
             50 hits                        7 hits
                |                             |
                +--------------+--------------+
                               v
                             FUSE                               3. merge rank lists
                            30 ids                                 (RRF, rank only)
                               v
                           HYDRATE                              4. fetch real text
                         30 passages                               ~2.6s
                               v
                            RERANK                              5. cross-encoder
                          6 passages                               scores, keeps best
                               v
                        back to the model

Every step runs on every search. Nothing here is optional or chosen between: the
only choice is upstream, where the model decides whether to call search_filings.
Two pairs run in PARALLEL, which is not the same as being alternatives; step 3
needs both results.


## Why two searches instead of one
Vector search and keyword search fail in opposite directions, so running both and
merging beats either alone.

    semantic (pgvector)   finds meaning, misses exact strings
                          "cloud margins" -> finds Azure profitability prose
                          but can miss the literal ticker or tag name
    keyword  (Postgres FTS)  finds exact strings, misses meaning
                          "PaymentsToAcquirePropertyPlantAndEquipment" lands
                          exactly, but a paraphrase finds nothing

## How the two ranked lists are merged
Reciprocal Rank Fusion. Each list votes with 1/(k + rank), and the votes are
summed. RRF uses only RANK, never the raw scores, which is the point: a cosine
distance and a ts_rank_cd score are not comparable numbers, and normalising them
against each other is guesswork. A chunk that both methods rank highly wins; a
chunk only one method loves can still surface.

## The pipeline
Every step runs on every search. Nothing here is optional or chosen between: the
only choice is upstream, where the model decides whether to call search_filings
at all.

    1. EMBED + KEYWORDS   both at once, in threads (independent, both are I/O)
    2. DUAL SEARCH        semantic and full-text, concurrently, candidate_k each
    3. FUSE               RRF over the two rank lists, keep a pool
    4. HYDRATE            load chunk rows, attach neighbours for context
    5. RERANK             cross-encoder scores the pool, keep top_k

Steps 1 and 2 are parallel because each is a network round trip and neither needs
the other's answer. Parallel, not alternative: step 3 needs both results.

## A real trace
Question: "What does Microsoft say about datacenter capacity constraints?"

    1. embedding : 1536 floats  [0.012, 0.052, 0.091, ...]
       keywords  : "datacenter capacity constraints"
                   ^ the question scaffolding ("What does Microsoft say about")
                     is stripped, because it would pollute a text search

    2. semantic  : 50 hits, best cosine  0.513
       full-text :  7 hits, best ts_rank 0.013
                   ^ semantic ALWAYS returns 50, since something is always
                     nearest in vector space. Full-text returned 7 because only
                     7 chunks literally contain those words.

    3. fused top-6, and where each came from:

           RRF      semantic_rank   fts_rank
           0.0318         1             5
           0.0312         4             4
           0.0306         9             2
           0.0299        14             1   <- semantic buried it, FTS loved it

       Only 7 of each method's top 20 overlapped. 13 of 20 were found by one
       method alone, which is the entire justification for searching twice.

    4. hydrate 30 chunks: ~2.6s, the second-largest cost in a search

    5. rerank: 30 candidates scored, 6 survive

## Funnel
    50 + 7 found  ->  30 fused  ->  6 returned
Early stages cast wide on purpose so the later, smarter ones have real choices.

## Neighbours
Each surviving chunk carries its immediate neighbours, because a table row or a
sentence often needs the line above it to be readable. Neighbours are recorded in
the TurnRegistry too, so the model may cite them; they are deduplicated against
chunks already returned so nothing appears twice.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import settings
from app.database.documents import get_chunks_by_ids, get_surrounding_chunks
from app.database.session import get_session
from app.retrieval.embeddings import embed_query
from app.retrieval.bm25 import score_rows as bm25_score_rows
from app.retrieval.hyde import expand as hyde_expand
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keywords import extract_fts_keywords
from app.retrieval.rerank import rerank
from app.retrieval.queries import full_text_candidates, full_text_search, semantic_search
from app.retrieval.types import RankedChunkHit, RetrievedPassage, SearchFilters

from app.database.models import DocumentChunk, SourceDocument


class DocumentRetriever:
    def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        include_neighbors: bool = True,
        session: Session | None = None,
    ) -> list[RetrievedPassage]:
        resolved_top_k = top_k if top_k is not None else settings.retrieval_top_k
        resolved_candidate_k = (
            candidate_k if candidate_k is not None else settings.retrieval_candidate_k
        )
        if session is not None:
            return self._search_with_session(
                session,
                query,
                filters=filters,
                top_k=resolved_top_k,
                candidate_k=resolved_candidate_k,
                include_neighbors=include_neighbors,
            )

        with get_session() as owned_session:
            return self._search_with_session(
                owned_session,
                query,
                filters=filters,
                top_k=resolved_top_k,
                candidate_k=resolved_candidate_k,
                include_neighbors=include_neighbors,
            )

    def _search_with_session(
        self,
        session: Session,
        query: str,
        *,
        filters: SearchFilters | None,
        top_k: int,
        candidate_k: int,
        include_neighbors: bool,
    ) -> list[RetrievedPassage]:
        # -- 1. EMBED + KEYWORDS ------------------------------------------
        # Independent of each other and both I/O bound, so they overlap.
        #   "What does Microsoft say about datacenter capacity constraints?"
        #     -> query_vec : 1536 floats, the question as a point in meaning-space
        #     -> fts_query : "datacenter capacity constraints"
        # Two readings of one question, because the two searches below need
        # different things. ~0.9s each; run together they cost 0.9s, not 1.8s.
        with ThreadPoolExecutor(max_workers=2) as prep:
            # HyDE feeds the vector leg only; full-text still matches the
            # user's literal words. See retrieval/hyde.py.
            embed_future = prep.submit(
                lambda: embed_query(hyde_expand(query)))
            kw_future = prep.submit(extract_fts_keywords, query, filters=filters)
            query_vec = embed_future.result()
            fts_query = kw_future.result()

        # -- 2. DUAL SEARCH -----------------------------------------------
        # candidate_k per method, wider than top_k: fusion needs enough
        # candidates to have something to disagree about.
        #   semantic  -> 50 hits, best cosine  0.513   (meaning; misses exact strings)
        #   full-text ->  7 hits, best ts_rank 0.013   (exact strings; misses meaning)
        # The counts differ because semantic always returns candidate_k (something
        # is always nearest), while full-text returns only genuine word matches.
        semantic_hits, fts_hits = _dual_search(
            query_vec,
            fts_query,
            candidate_k=candidate_k,
            filters=filters,
        )

        # -- 3. FUSE --------------------------------------------------------
        # Only the ORDER of each list is used. Raw scores from the two methods
        # are on incomparable scales and are deliberately discarded here.
        # RRF = sum of 1/(k + rank) across lists. Observed on the query above:
        #     RRF 0.0318  semantic_rank=1   fts_rank=5
        #     RRF 0.0299  semantic_rank=14  fts_rank=1  <- FTS rescued it
        # Only 7 of each method's top 20 overlapped, so 13 were found by one
        # method alone. That is the whole reason for searching twice.
        semantic_ids = [hit.chunk_id for hit in semantic_hits]
        fts_ids = [hit.chunk_id for hit in fts_hits]
        # Keep a POOL, not the final cut. Fusion ranks by position and cannot tell
        # a direct answer from a topically-similar paragraph, so handing it the
        # final say wastes the reranker before it runs. The cross-encoder below
        # picks top_k out of this pool by actually reading each one.
        pool = max(top_k, settings.retrieval_rerank_pool) if settings.retrieval_rerank_enabled else top_k
        fused = reciprocal_rank_fusion(
            [semantic_ids, fts_ids],
            k=settings.retrieval_rrf_k,
        )[:pool]

        if not fused:
            return []

        # -- 4. HYDRATE -----------------------------------------------------
        # Searches returned ids and scores only. Now fetch the actual text and
        # document metadata, and attach neighbouring chunks for readability.
        # Measured at ~2.6s for 30 chunks: the second-largest cost in a search,
        # and the reason the pool size is a real dial and not just a quality one.
        fused_ids = [chunk_id for chunk_id, _ in fused]
        fusion_scores = {chunk_id: score for chunk_id, score in fused}
        chunks_by_id = get_chunks_by_ids(session, fused_ids)

        passages: list[RetrievedPassage] = []
        seen_neighbor_ids: set[UUID] = set(fused_ids)

        for chunk_id in fused_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or chunk.document is None:
                continue

            neighbors: list[RetrievedPassage] = []
            if include_neighbors:
                for neighbor_chunk in get_surrounding_chunks(
                    session,
                    chunk_id,
                    settings.retrieval_neighbor_radius,
                ):
                    if neighbor_chunk.id in seen_neighbor_ids:
                        continue
                    if neighbor_chunk.document is None:
                        continue
                    seen_neighbor_ids.add(neighbor_chunk.id)
                    neighbors.append(
                        _passage_from_chunk(
                            neighbor_chunk,
                            neighbor_chunk.document,
                            fusion_score=0.0,
                        )
                    )

            passages.append(
                _passage_from_chunk(
                    chunk,
                    chunk.document,
                    fusion_score=fusion_scores[chunk_id],
                    neighbors=neighbors,
                )
            )

        # -- 5. RERANK -------------------------------------------------------
        # Query and passage are read together here, which is what makes a poor
        # match visible as a low score instead of just a lower rank.
        #   30 candidates in -> 6 out
        #   present topic ->  +5.2 ("AI datacenter investment" in MSFT)
        #   absent  topic ->  -4.7 ("why Apple spends on capex", which AAPL
        #                            never explains) -> the model can say so
        # Fusion cannot do this: every RRF score above is ~0.03 whether the
        # passage is a bullseye or merely the nearest thing in the corpus.
        return rerank(query, passages, top_k=top_k)


def _dual_search(
    query_vec: list[float],
    fts_query: str,
    *,
    candidate_k: int,
    filters: SearchFilters | None,
) -> tuple[list[RankedChunkHit], list[RankedChunkHit]]:
    """Run semantic and full-text search in parallel (separate DB sessions)."""

    def semantic() -> list[RankedChunkHit]:
        with get_session() as search_session:
            return semantic_search(
                search_session,
                query_vec,
                limit=candidate_k,
                filters=filters,
            )

    def fts() -> list[RankedChunkHit]:
        with get_session() as search_session:
            if not settings.retrieval_bm25_enabled:
                return full_text_search(
                    search_session, fts_query, limit=candidate_k, filters=filters,
                )
            # Pull a wide set from the index WITH its tsvectors, then let BM25
            # decide the order. ts_rank_cd has no IDF, so its own top-50 buries
            # the rare terms that make a lexical match worth having; rescoring
            # can only promote what it was handed, hence the wider pull. One
            # query rather than two: the vectors come back with the hits, which
            # removes the round trip that was 83% of BM25's cost.
            rows = full_text_candidates(
                search_session,
                fts_query,
                limit=settings.retrieval_bm25_candidates,
                filters=filters,
            )
            ordered = bm25_score_rows(
                search_session, fts_query, rows, top_k=candidate_k,
            )
            return [
                RankedChunkHit(chunk_id=chunk_id, rank=position, score=0.0)
                for position, chunk_id in enumerate(ordered, 1)
            ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(semantic)
        fts_future = executor.submit(fts)
        return semantic_future.result(), fts_future.result()


def _passage_from_chunk(
    chunk: DocumentChunk,
    document: SourceDocument,
    *,
    fusion_score: float,
    neighbors: list[RetrievedPassage] | None = None,
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        page=chunk.page,
        section=chunk.section,
        fusion_score=fusion_score,
        ticker=document.ticker,
        company_name=document.company_name,
        form=document.form,
        filing_date=document.filing_date,
        fiscal_year=document.fiscal_year,
        accession_number=document.accession_number,
        neighbors=neighbors or [],
    )
