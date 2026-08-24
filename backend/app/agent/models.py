"""## The models a question can be answered with

## Why this is a choice and not a setting
Measured on this project, capability and cost do not trade off the way people
expect. On the same DCF question, flash-lite spent 8,167 tokens and the reasoning
model spent 38,605. The expensive model is not slower to converge; it simply
thinks more per answer, and produces a better one.

What that buys, concretely: on that DCF, flash-lite skipped the sensitivity table
and the comparison to market price even though the loaded skill mandates both.
The stronger model produced both unprompted and identified the peak-capex problem
on its own.

So the honest split is:

    lookups, comparisons, narrative retrieval   ->  flash-lite is fine
    valuation, multi-step judgement             ->  worth paying for

Default stays flash-lite because most questions here are the first kind.

## Adding one
Append to MODELS. Anything the Gemini API lists under generateContent works; the
agent is rebuilt per model and cached, so switching costs one construction and
nothing after that.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelChoice:
    id: str
    label: str
    hint: str
    default: bool = False


# Ordered cheapest to most capable, which is also the order the UI shows.
MODELS: list[ModelChoice] = [
    ModelChoice(
        id="gemini-3.1-flash-lite",
        label="Flash Lite",
        hint="Fastest and cheapest. Fine for figures, comparisons and filing text.",
        default=True,
    ),
    ModelChoice(
        id="gemini-3.7-flash",
        label="Flash 3.7",
        hint="Newer mid tier. Better judgement, still quick.",
    ),
    ModelChoice(
        id="gemini-3.1-pro-preview",
        label="Pro",
        hint="Strongest reasoning. Use for valuation and multi-step analysis; costs several times more per answer.",
    ),
]

_BY_ID = {m.id: m for m in MODELS}
DEFAULT_MODEL = next(m.id for m in MODELS if m.default)


def resolve(model_id: str | None) -> str:
    """Return a known model id, falling back to the default.

    Unknown ids fall back rather than raising: a stale value in a browser tab
    should not make the chat unusable.
    """
    if model_id and model_id in _BY_ID:
        return model_id
    return DEFAULT_MODEL
