"""## Which tools are worth describing for THIS question

## Why this exists
Tool schemas are re-sent on every round, not once per question. Measured here:

    12 tool schemas       2,707 tokens   per request
      run_dcf_valuation     689
      run_comps_valuation   648          <- 49% of the total, between them
    instructions.md       1,270
    ----------------------------------
    base per request      3,980

On "what was Apple's capital expenditure in fiscal 2025?" that base is paid three
times, so 65% of an 18,435-token answer was fixed overhead, and 4,011 tokens of it
described DCF and comps engines the question never touched.

## How it decides
Valuation tools appear only when the question sounds like valuation. The vocabulary
below is deliberately WIDE: a false positive costs a few hundred tokens, while a
false negative means the model cannot value a company at all. When in doubt the
tools stay.

Fundamentals and market tools are never gated. They are small, and any question
might need them.

## Retrieval is gated differently, by MODE rather than by question
There are two ways to find filing text and they are alternatives, not a menu:
RAG searches the ingested index, agentic retrieval searches SEC directly. Showing
both at once would let the model mix them, double the schema cost, and produce
answers whose provenance depends on which tool it happened to reach for. So the
selected mode's tools are visible and the other mode's are hidden entirely.
"""

from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from app.retrieval import modes

# Only these two are ever hidden; together they are half the schema budget.
_VALUATION_TOOLS = {
    "run_dcf_valuation",
    "run_comps_valuation",
    "get_comps_inputs",
    "reason_about_assumptions",
}

_VALUATION_WORDS = (
    "valu",           # value, valuation, valuing, undervalued, fair value
    "dcf", "discounted cash flow", "intrinsic", "worth", "fair price",
    "comps", "comparable", "multiple", "ev/", "p/e", "price target",
    "wacc", "terminal growth", "cost of capital", "cost of equity",
    "overvalued", "undervalued", "expensive", "cheap", "what should",
)


def _question_of(ctx: RunContext) -> str:
    prompt = getattr(ctx, "prompt", None)
    if isinstance(prompt, str):
        return prompt.lower()
    if isinstance(prompt, list):  # multi-part prompt
        return " ".join(str(p) for p in prompt).lower()
    return ""


#: Shared with agent.py, which uses the same vocabulary to decide whether a
#: question is worth thinking about. One list, so the tool gate and the
#: reasoning default can never disagree about what counts as a valuation.
VALUATION_WORDS = _VALUATION_WORDS


def wants_valuation(ctx: RunContext) -> bool:
    return any(word in _question_of(ctx) for word in _VALUATION_WORDS)


#: Every tool owned by some retrieval mode. A tool in this set is shown only
#: when its own mode is the selected one.
_MODE_TOOLS = modes.retrieval_tool_names()


def include_tool(ctx: RunContext, tool: ToolDefinition) -> bool:
    """Filter passed to FilteredToolset: True keeps the tool visible."""
    if tool.name in _MODE_TOOLS:
        selected = getattr(getattr(ctx, "deps", None), "retrieval_mode", None)
        return tool.name in modes.mode(selected).tools
    if tool.name not in _VALUATION_TOOLS:
        return True
    return wants_valuation(ctx)
