"""Structured output types for the document agent."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_index: int = Field(
        description="1-based index referenced as [n] in the answer text"
    )
    chunk_id: UUID = Field(description="UUID of the cited document chunk")
    excerpt: str = Field(
        description="Verbatim substring from the chunk text supporting the claim"
    )


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Plain-English answer with [n] citation markers")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations backing factual claims in the answer",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True when the corpus does not contain enough evidence to answer",
    )
