"""Pydantic models shared by retrieval and future agent tools."""

from __future__ import annotations

from datetime import date
from uuid import UUID

from pydantic import BaseModel, Field

MAX_PASSAGE_EXCERPT_CHARS = 800
MAX_AGENT_OUTPUT_CHARS = 12_000


class SearchFilters(BaseModel):
    ticker: str | None = None
    fiscal_years: list[int] | None = None
    form: str | None = None


class RankedChunkHit(BaseModel):
    chunk_id: UUID
    rank: int
    score: float | None = None


class RetrievedPassage(BaseModel):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    text: str
    page: str | None
    section: str | None
    fusion_score: float
    ticker: str
    company_name: str | None
    form: str
    filing_date: date
    fiscal_year: int | None
    accession_number: str
    # None when reranking is off or unavailable; a raw cross-encoder logit
    # otherwise, roughly -11 (irrelevant) to +11 (directly answers).
    rerank_score: float | None = None
    neighbors: list[RetrievedPassage] = Field(default_factory=list)


def _format_one_passage(passage: RetrievedPassage, *, include_neighbors: bool) -> str:
    year = passage.fiscal_year or passage.filing_date.year
    page = f" p.{passage.page}" if passage.page else ""
    section = f" ({passage.section})" if passage.section else ""
    excerpt = passage.text.strip()
    if len(excerpt) > MAX_PASSAGE_EXCERPT_CHARS:
        excerpt = excerpt[:MAX_PASSAGE_EXCERPT_CHARS] + "..."
    # chunk_id is deliberately NOT wrapped in square brackets. Square brackets in
    # an answer mean a citation index, and the grounding validator rejects any
    # answer whose markers are not small integers. Showing the id as "[uuid]" here
    # taught the model to copy that shape into its prose, which failed validation
    # and forced a full re-run of the agent.
    rel = f" relevance={passage.rerank_score}" if passage.rerank_score is not None else ""
    header = (
        f"{passage.ticker} {passage.form} FY{year}{page}{section}{rel} "
        f"chunk_id={passage.chunk_id}: {excerpt}"
    )
    lines = [header]
    if include_neighbors:
        for neighbor in passage.neighbors:
            neighbor_excerpt = neighbor.text.strip()
            if len(neighbor_excerpt) > MAX_PASSAGE_EXCERPT_CHARS:
                neighbor_excerpt = neighbor_excerpt[:MAX_PASSAGE_EXCERPT_CHARS] + "..."
            lines.append(
                f"  neighbor idx={neighbor.chunk_index} "
                f"chunk_id={neighbor.chunk_id}: {neighbor_excerpt}"
            )
    return "\n".join(lines)


def format_passages_for_agent(passages: list[RetrievedPassage]) -> str:
    """Bounded, grep-style text for PydanticAI tool responses."""
    if not passages:
        return "No matching passages found in the filing corpus."

    blocks = [_format_one_passage(p, include_neighbors=True) for p in passages]
    output = "\n\n".join(blocks)
    if len(output) > MAX_AGENT_OUTPUT_CHARS:
        output = (
            output[:MAX_AGENT_OUTPUT_CHARS]
            + f"\n... truncated to {len(passages)} passages."
        )
    return output
