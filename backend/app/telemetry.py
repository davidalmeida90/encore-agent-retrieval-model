"""Per-turn LLM call accounting.

The repo already computes `requests` and `tool_calls` inside the agent and hands
them to `report_progress`, which the server never listens to. Two more Gemini
calls live outside the agent entirely and are invisible to `usage.requests`:

  * extract_fts_keywords  — one per search_filings
  * GroundingValidator    — one per turn, batched over citations

This module counts all of them against a per-turn context, so a turn can report
what it actually spent rather than what the agent alone saw.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)


@dataclass
class TurnCounters:
    agent_requests: int = 0
    tool_calls: int = 0
    gemini_keyword_calls: int = 0
    gemini_validator_calls: int = 0
    openai_embedding_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    events: list[str] = field(default_factory=list)

    @property
    def gemini_total(self) -> int:
        """Every Gemini call, including the two the agent cannot see."""
        return (
            self.agent_requests
            + self.gemini_keyword_calls
            + self.gemini_validator_calls
        )


_current: contextvars.ContextVar[TurnCounters | None] = contextvars.ContextVar(
    "turn_counters", default=None
)


def start_turn() -> TurnCounters:
    counters = TurnCounters()
    _current.set(counters)
    return counters


def current() -> TurnCounters | None:
    return _current.get()


def _bump(field_name: str, amount: int = 1) -> None:
    counters = _current.get()
    if counters is not None:
        setattr(counters, field_name, getattr(counters, field_name) + amount)


def record_keyword_call() -> None:
    _bump("gemini_keyword_calls")


def record_validator_call() -> None:
    _bump("gemini_validator_calls")


def record_embedding_call() -> None:
    _bump("openai_embedding_calls")


def progress_listener(message: str) -> None:
    """Attach to app.agent.progress so agent-side numbers stop vanishing."""
    counters = _current.get()
    if counters is None:
        return
    counters.events.append(message)
    if message.startswith("agent run done"):
        for token in message.split():
            if "=" not in token:
                continue
            key, _, raw = token.partition("=")
            if key in {"requests", "tool_calls", "input_tokens", "output_tokens"}:
                try:
                    value = int(raw)
                except ValueError:
                    continue
                setattr(
                    counters,
                    "agent_requests" if key == "requests" else key,
                    value,
                )


def log_turn(counters: TurnCounters, *, thread_id: str, query: str) -> None:
    log.info(
        "turn_cost",
        thread_id=thread_id,
        query=query[:80],
        gemini_total=counters.gemini_total,
        agent_requests=counters.agent_requests,
        tool_calls=counters.tool_calls,
        keyword_calls=counters.gemini_keyword_calls,
        validator_calls=counters.gemini_validator_calls,
        openai_embeddings=counters.openai_embedding_calls,
        input_tokens=counters.input_tokens,
        output_tokens=counters.output_tokens,
    )
    # plain line too, so it is readable in the uvicorn console
    print(
        f"[turn cost] gemini={counters.gemini_total} "
        f"(agent={counters.agent_requests} keyword={counters.gemini_keyword_calls} "
        f"validator={counters.gemini_validator_calls}) "
        f"tools={counters.tool_calls} openai_embed={counters.openai_embedding_calls} "
        f"tokens_in={counters.input_tokens} tokens_out={counters.output_tokens}",
        flush=True,
    )


# --------------------------------------------------------------- daily quota
# Nothing above survives a single turn, so the daily free-tier allowance drained
# invisibly until requests started failing. SpendLimits calls record_spend after
# every model response; this turns that into a visible running count.
_LAST_DAILY: dict[str, object] = {}


def record_spend(snapshot: object) -> None:
    """Log what each budget has left. Wired as SpendLimits(on_spend=...).

    Never raises: accounting must not be able to break a working answer.
    """
    try:
        for status in getattr(snapshot, "budgets", []) or []:
            name = getattr(getattr(status, "budget", None), "name", "?")
            spent = getattr(status, "spent", None)
            remaining = getattr(status, "remaining_usd", None)
            if name == "per-question-tokens" and spent is not None:
                # running token total for THIS question, so a runaway is visible
                # while it happens rather than only in the exception at the end
                log.info("question_tokens", used=int(getattr(spent, "tokens", 0) or 0))
            if name == "daily-requests" and remaining is not None:
                used = int(getattr(spent, "requests", 0) or 0)
                left = int(remaining)
                _LAST_DAILY.update({"used": used, "remaining": left})
                if getattr(status, "exhausted", False):
                    log.error("daily_quota_exhausted", used=used,
                              hint="free tier resets at midnight Pacific")
                elif getattr(status, "warning", False):
                    log.warning("daily_quota_low", used=used, remaining=left)
                else:
                    log.info("daily_quota", used=used, remaining=left)
    except Exception:  # pragma: no cover - accounting must never break a turn
        log.debug("record_spend_failed", exc_info=True)


def daily_usage() -> dict[str, object]:
    """Last known daily request count, for a status endpoint or a CLI check."""
    return dict(_LAST_DAILY)
