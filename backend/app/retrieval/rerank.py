"""## Cross-encoder reranking: the step that can tell "close" from "answers"

## Why fusion alone is not enough
RRF ranks by POSITION in two lists, so its scores carry no relevance information.
Measured on this corpus, the top six fused scores were 0.0164, 0.0161, 0.0159,
0.0156, 0.0154, 0.0152: a perfect match and a hopeless one look identical.

Raw embedding similarity is no better as a stopping signal, because a bi-encoder
compares two vectors made independently and therefore measures TOPICAL closeness,
not whether a passage answers anything. Calibrated against known cases:

    PRESENT  MSFT  AI datacenter investment        0.602
    PRESENT  AAPL  segment results                 0.643
    PRESENT  AAPL  risk factors competition        0.482
    ABSENT   AAPL  why capex at this level         0.556   <- beats a PRESENT case
    ABSENT   AAPL  GPU purchase commitments        0.426
    ABSENT   MSFT  iPhone unit sales by colour     0.368

The ranges overlap, so no threshold separates them. "Why does Apple spend this
much" scores 0.556 against depreciation and PP&E notes that never say why: the
topic is all over the filing, the answer is not in it.

## What a cross-encoder does differently
It reads the question and the passage TOGETHER in one forward pass, so it scores
whether this passage answers this question rather than whether they are about the
same subject. That is exactly the distinction the numbers above fail to make, and
it is why absence becomes detectable.

Model is ms-marco-MiniLM-L6-v2: 23M parameters, ~90MB, 0.41s for 50 pairs on CPU.
Small enough that no GPU is needed, which keeps torch on the CPU build that the
docling ingest pipeline depends on.

Scores are raw logits, not probabilities: roughly -11 (irrelevant) to +11 (a direct
answer), centred near 0. Compare them to each other, not to a fixed scale.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from app.retrieval.types import RetrievedPassage

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    """Load once per process. First call downloads ~90MB, then it is cached."""
    import os

    # Must be set BEFORE torch is imported. Torch allocates a per-thread arena,
    # and with the app already holding SQLAlchemy, Supabase and the OpenAI client
    # this once exhausted the Windows page file:
    #   OSError: The paging file is too small for this operation to complete
    # Root cause was the disk at 0.6% free, not thread count. 8 threads now load
    # cleanly inside the running app and cut reranking roughly 4x. Tunable via
    # retrieval_rerank_threads if the disk fills again.
    threads = str(settings.retrieval_rerank_threads)
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(settings.retrieval_rerank_threads)
    name = settings.retrieval_rerank_model
    tokenizer = AutoTokenizer.from_pretrained(name)
    # low_cpu_mem_usage streams the weights in rather than memory-mapping the whole
    # checkpoint at once. Without it, loading inside the running app (which already
    # holds torch, docling and SQLAlchemy) fails on Windows with
    # "The paging file is too small for this operation to complete".
    model = AutoModelForSequenceClassification.from_pretrained(
        name, low_cpu_mem_usage=True
    ).eval()

    # Device choice. The GPU matters less than it looks: after multi-threading
    # the CPU, reranking is ~1.6s of a ~5.7s search, the rest being network and
    # database. Running on the GPU is mainly about leaving the 16 CPU cores free
    # for other work rather than about the seconds saved.
    want = settings.retrieval_rerank_device
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        try:
            model = model.to("cuda")
            device = "cuda"
        except Exception:
            log.warning("could not move reranker to GPU, staying on CPU", exc_info=True)
            device = "cpu"
    else:
        device = "cpu"

    log.info("reranker ready on %s", device)
    return tokenizer, model, torch, device


def rerank(
    query: str,
    passages: list[RetrievedPassage],
    *,
    top_k: int,
) -> list[RetrievedPassage]:
    """Score every passage against the query, return the best `top_k`.

    Failure is never fatal: if the model cannot load, the fused order is returned
    unchanged. A slower, blunter answer beats no answer.
    """
    if not passages or not settings.retrieval_rerank_enabled:
        return passages[:top_k]

    try:
        tokenizer, model, torch, device = _model()
        texts = [p.text for p in passages]
        with torch.no_grad():
            batch = tokenizer(
                [query] * len(texts),
                texts,
                padding=True,
                truncation=True,
                max_length=settings.retrieval_rerank_max_tokens,
                return_tensors="pt",
            )
            if device == "cuda":
                batch = {k: v.to("cuda") for k, v in batch.items()}
            scores = model(**batch).logits.squeeze(-1).float().cpu().tolist()
        if isinstance(scores, float):  # single passage squeezes to a scalar
            scores = [scores]
    except Exception:
        log.warning("rerank unavailable, falling back to fusion order", exc_info=True)
        return passages[:top_k]

    scored = [
        p.model_copy(update={"rerank_score": round(float(s), 3)})
        for p, s in zip(passages, scores, strict=True)
    ]
    scored.sort(key=lambda p: p.rerank_score or 0.0, reverse=True)
    return scored[:top_k]


def warm_reranker() -> None:
    """Load the model now so the first real question does not pay for it."""
    _model()
