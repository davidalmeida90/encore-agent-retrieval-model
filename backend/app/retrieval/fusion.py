"""Reciprocal Rank Fusion for hybrid retrieval."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID


def reciprocal_rank_fusion(
    rankings: list[list[UUID]],
    *,
    k: int = 60,
) -> list[tuple[UUID, float]]:
    scores: dict[UUID, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])
