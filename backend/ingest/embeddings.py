"""## Embedding: turning chunk text into a searchable vector

    chunk text  ->  OpenAI text-embedding-3-small  ->  1536 floats  ->  pgvector

## What an embedding is, in one line
A point in a 1536-dimensional space, positioned so that text about similar things
lands nearby. Search then becomes geometry: embed the question, find the nearest
chunks by cosine distance.

## Why 1536 dimensions
That is `text-embedding-3-small`'s native size, and the number is baked into
three places that must agree or nothing works:

    settings.openai_embedding_dimensions   = 1536
    the `vector(1536)` column in Postgres   (Alembic migration)
    the HNSW index built over that column

A mismatch is not a soft failure. Postgres rejects the insert, which is the
correct behaviour: half a corpus at the wrong width would silently return
nonsense rather than erroring.

## Why batches of 100
One HTTP round trip per chunk would make ingesting a filing 700 requests. The API
accepts a list, so chunks go up in batches and the cost is a handful of calls per
document. Results come back with an `index` field, and the code re-sorts by it
rather than trusting arrival order: a batch returned out of order would attach
every embedding to the wrong chunk, and every later search would be subtly,
silently wrong.

## What this costs
`text-embedding-3-small` is roughly $0.02 per million tokens. A 10-K is about
700 chunks of ~400 tokens, so ~280,000 tokens, or well under a cent per filing.
Embedding is never the expensive part of ingestion; docling conversion is.

## Why the same model must be used at query time
`app/retrieval/embeddings.py` embeds the QUESTION with this same model. Two
models produce two incompatible coordinate systems, so mixing them would compare
points in unrelated spaces and return confident nonsense. Changing the embedding
model means re-embedding the entire corpus, never just the queries.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import settings

EMBED_BATCH_SIZE = 100


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], *, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    if not texts:
        return []

    expected_dims = settings.openai_embedding_dimensions
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = _client().embeddings.create(
            input=batch,
            model=settings.openai_embedding_model,
            dimensions=expected_dims,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        for item in ordered:
            embedding = item.embedding
            if len(embedding) != expected_dims:
                raise ValueError(
                    f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
                )
            vectors.append(embedding)

    return vectors
