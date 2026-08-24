"""OpenAI query embedding for live retrieval."""

from __future__ import annotations

from openai import OpenAI

from app import telemetry
from app.config import settings


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def embed_query(text: str) -> list[float]:
    telemetry.record_embedding_call()
    response = _client().embeddings.create(
        input=[text],
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
    )
    embedding = response.data[0].embedding
    expected_dims = settings.openai_embedding_dimensions
    if len(embedding) != expected_dims:
        raise ValueError(
            f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
        )
    return embedding
