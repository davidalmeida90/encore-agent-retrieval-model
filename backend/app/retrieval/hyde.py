"""Embed a hypothetical answer, not the question.

## The problem it solves
A question and the passage that answers it are written in different registers.
"How does Apple describe its dependence on China?" shares almost no vocabulary
with "Substantially all of the Company's manufacturing is performed by
outsourcing partners located primarily in China mainland". The embedding has to
bridge that gap unaided, and measurably struggles: on the retrieval eval, dense
recall was 62% for questions using the filing's own wording and 31% for the same
questions paraphrased.

HyDE closes the gap from the other side. A cheap model writes what the answer
would probably look like, and that text is embedded instead. A fabricated filing
paragraph sits far closer to a real one than any question does, and it does not
matter that the specifics are invented: the vector is used only to find
neighbours, and the answer the user sees is still built from retrieved text.

Measured on the same 52 questions, question-only against question-plus-hypothesis:

    verbatim      62% -> 77%
    paraphrased   31% -> 38%
    overall       46% -> 58%

## Two deliberate choices
The question is KEPT alongside the hypothesis rather than replaced by it, so its
own rare terms (a ticker, a segment name) still pull on the vector.

This feeds the DENSE leg only. Full-text search matches literal tokens, and
invented sentences would put fabricated words into a lexical query, which is the
one place a hallucination could actually mislead retrieval.
"""

from __future__ import annotations

import logging

from app.config import settings

log = logging.getLogger(__name__)

_PROMPT = (
    "Write one paragraph that could plausibly appear in an SEC 10-K and would "
    "answer this question. Use the formal register of a filing. Invent specifics "
    "if needed; accuracy does not matter, only that it reads like a filing.\n\n"
    "Question: "
)


def expand(question: str) -> str:
    """Question plus a hypothetical answer, or just the question if that fails.

    Fails open on purpose. A retrieval that is 12 points worse is a bad day; a
    retrieval that raises because a side model was slow is a broken product.
    """
    if not settings.retrieval_hyde_enabled or not settings.gemini_api_key:
        return question
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.gemini_api_key,
                        base_url=settings.gemini_openai_base_url,
                        timeout=settings.retrieval_hyde_timeout_seconds)
        completion = client.chat.completions.create(
            model=settings.gemini_keyword_model,
            messages=[{"role": "user", "content": _PROMPT + question}],
            temperature=0,
            max_tokens=220,
        )
        hypothesis = (completion.choices[0].message.content or "").strip()
    except Exception:
        log.warning("HyDE expansion failed, embedding the question alone", exc_info=True)
        return question
    return f"{question}\n{hypothesis}" if hypothesis else question
