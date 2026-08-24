"""## Where a question's tokens actually went

## Why there is an "unattributed" bucket
Providers report ONE input-token total per request, never a breakdown. Components
here are counted locally with tiktoken; whatever is left over goes to
`unattributed` (the GroundedAnswer JSON schema, per-message framing, role markers,
provider-side wrapping).

An earlier version SCALED the measured parts up to the provider's total instead.
It summed neatly and it lied: on a real question it reported the system prompt at
80% of input when the true figure was 45%, because everything unmeasured was
silently redistributed into what was measured. A tuning decision was nearly made
on that number. A visible remainder beats a tidy wrong one.

## What gets attributed
    system prompt   instructions.md + one line per skill. Fixed, re-sent EVERY round.
    tool schemas    names, descriptions and JSON schemas of the offered tools.
                    Also re-sent every round, whether called or not.
    conversation    the question, and the model's own replies.
    tool results    what the tools returned. Persist in history, so a big return
                    is paid for again on every later round.
    output          tokens the model generated.

## The thing this makes visible
Cost grows with the SQUARE of the round count, because round 6 re-sends rounds
1-5. Measured on real runs:

    3 requests ->  18,435 tokens ->  6,145 per request
    10 requests -> 174,269 tokens -> 17,427 per request

So a breakdown showing "system prompt + tool schemas = 65% of a simple question"
is not a rounding detail. It is the reason short questions cost what they do, and
why hiding the two valuation tools on non-valuation questions saved ~4,000 tokens.

## Calls outside the agent loop
The keyword extractor (one per search) and the grounding judge (one per turn) are
separate model calls that `usage.input_tokens` never sees. They are counted here
from telemetry so the reported total matches what the provider actually billed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any


@dataclass
class UsageBreakdown:
    """One question's token spend, split by what the tokens were spent on."""

    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # estimated components of input_tokens, scaled to sum to it
    system_prompt: int = 0
    tool_schemas: int = 0
    conversation: int = 0
    tool_results: int = 0

    # everything measured above subtracted from the provider's real total:
    # output schema, message framing, role markers, provider-side prompt wrapping
    unattributed: int = 0

    # separate model calls the agent's own usage never sees
    keyword_calls: int = 0
    validator_calls: int = 0

    per_tool: dict[str, int] = field(default_factory=dict)

    # The literal text behind the two fixed components, so the UI can show what
    # is actually being paid for on every round rather than just a number.
    system_prompt_text: str = ""
    tool_schemas_text: str = ""

    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _tok(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_encoder().encode(text))
    except Exception:
        # 4 chars per token is the usual rough ratio; never fail a turn over
        # accounting.
        return max(1, len(text) // 4)


def _schema_text(offered: set[str]) -> str:
    """Reconstruct what the tools cost, as readable text, for the UI to show."""
    import inspect

    lines: list[str] = []
    try:
        from app.agent.tools import TOOLS

        for fn in TOOLS:
            name = getattr(fn, "__name__", "")
            if offered and name not in offered:
                continue
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (fn.__doc__ or "").strip()
            lines.append(f"{name}{sig}\n    {doc}\n")
    except Exception:
        return ""
    return "\n".join(lines)


def build(result: Any, *, counters: Any = None) -> UsageBreakdown:
    """Attribute one completed run's tokens. Never raises."""
    out = UsageBreakdown()
    try:
        usage = result.usage
        out.requests = usage.requests or 0
        out.tool_calls = usage.tool_calls or 0
        out.input_tokens = usage.input_tokens or 0
        out.output_tokens = usage.output_tokens or 0
        out.total_tokens = out.input_tokens + out.output_tokens

        from app.agent.agent import INSTRUCTIONS
        from app.agent.tools import skill_descriptions

        messages = result.all_messages()
        offered: set[str] = set()
        conversation = 0
        tool_results = 0

        for message in messages:
            for part in getattr(message, "parts", []):
                kind = type(part).__name__
                if kind == "ToolCallPart":
                    name = getattr(part, "tool_name", "")
                    if name and name != "final_result":
                        offered.add(name)
                        conversation += _tok(str(getattr(part, "args", "")))
                elif kind == "ToolReturnPart":
                    name = getattr(part, "tool_name", "") or "unknown"
                    size = _tok(str(getattr(part, "content", "")))
                    tool_results += size
                    out.per_tool[name] = out.per_tool.get(name, 0) + size
                elif kind in {"UserPromptPart", "TextPart"}:
                    conversation += _tok(str(getattr(part, "content", "")))

        # Fixed cost, re-sent on every request. This is the term that makes a
        # 2-round question cost more than people expect.
        skills = skill_descriptions()
        system_text = INSTRUCTIONS + "\n\n## Skills\n" + skills
        schema_text = _schema_text(offered)

        per_round = _tok(system_text)
        system = per_round * max(1, out.requests)
        schemas = _tok(schema_text) * max(1, out.requests)

        out.system_prompt_text = system_text
        out.tool_schemas_text = schema_text

        # Report what was MEASURED, and put the remainder in its own bucket.
        #
        # An earlier version scaled the components up to the provider's total.
        # That made them sum neatly but lied about the shares: everything not
        # measured here (the GroundedAnswer JSON schema, per-message framing,
        # role markers, provider-side wrapping) was silently redistributed into
        # the components that were. It inflated "system prompt" from a true ~45%
        # to a reported 80%, and a tuning decision was nearly made on that number.
        #
        # An honest unattributed bucket is more useful than a tidy one that is
        # wrong: a large remainder is itself a finding.
        out.system_prompt = system
        out.tool_schemas = schemas
        out.conversation = conversation
        out.tool_results = tool_results
        measured = system + schemas + conversation + tool_results
        out.unattributed = max(0, out.input_tokens - measured)

        if counters is not None:
            out.keyword_calls = getattr(counters, "gemini_keyword_calls", 0)
            out.validator_calls = getattr(counters, "gemini_validator_calls", 0)

        out.notes = (
            "Totals come from the provider and are exact. Components are counted "
            "locally with tiktoken, so they carry a small tokenizer difference. "
            "Unattributed is the remainder: output schema, message framing and "
            "provider-side wrapping, none of which is separately reported."
        )
    except Exception:
        out.notes = "Breakdown unavailable for this run."
    return out
