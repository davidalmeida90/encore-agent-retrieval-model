"""Shared safety rails for every tool in this package.

## Why these exist
Two different runaway modes were observed in real runs, and each wastes tokens in
a way the model cannot detect from inside the loop:

1. A tool that keeps FAILING. One broken comps call retried seven times and spent
   112,000 tokens in a single question.
2. A tool that keeps SUCCEEDING and returning the same answer. One run called
   get_sec_financials with byte-identical arguments seven times because the model
   disbelieved the number rather than because anything was wrong.

Guard 1 catches the first, guard 2 catches the second. Both return a STOP payload
telling the model to report what it has instead of calling again.
"""

from __future__ import annotations

import json
from dataclasses import asdict  # noqa: F401  (used by _safe)
from typing import Any

from app.agent import tuning


MAX_TOOL_FAILURES = tuning.MAX_TOOL_FAILURES

_failures: dict[tuple[str, str], int] = {}

def _fail_key(ctx: Any, tool: str) -> tuple[str, str]:
    return (str(getattr(ctx.deps, "thread_id", "?")), tool)

def _too_many_failures(ctx: Any, tool: str) -> str | None:
    if _failures.get(_fail_key(ctx, tool), 0) >= MAX_TOOL_FAILURES:
        return json.dumps({
            "error": f"STOP: {tool} has already failed {MAX_TOOL_FAILURES} times this turn.",
            "instruction": ("Do NOT call this tool again. Report to the user what you "
                            "were unable to compute and why, using the data you already "
                            "have. Retrying will not succeed."),
        })
    return None

def _record_failure(ctx: Any, tool: str) -> None:
    k = _fail_key(ctx, tool)
    _failures[k] = _failures.get(k, 0) + 1

MAX_IDENTICAL_CALLS = tuning.MAX_IDENTICAL_CALLS

_repeats: dict[tuple[str, str, str], int] = {}

def _too_many_repeats(ctx: Any, tool: str, args: Any) -> str | None:
    k = (str(getattr(ctx.deps, "thread_id", "?")), tool,
         json.dumps(args, sort_keys=True, default=str))
    _repeats[k] = _repeats.get(k, 0) + 1

    # A returned STOP string is only ADVICE, and advice can be ignored. Observed
    # on 2026-08-20: the guard returned STOP and the model issued the identical
    # call nine more times, each one cheap in itself but costing a full history
    # re-send per round. The run only ended when the token budget killed it.
    #
    # So the soft stop gets a few chances, then becomes a hard one. Raising ends
    # the run with a clear reason instead of letting it grind to the ceiling.
    if _repeats[k] > MAX_IDENTICAL_CALLS + 2:
        raise RuntimeError(
            f"{tool} was called {_repeats[k]} times with identical arguments "
            "after being told to stop. Aborting rather than looping to the token "
            "ceiling. The tool is fine; the model would not move on."
        )

    if _repeats[k] > MAX_IDENTICAL_CALLS:
        return json.dumps({
            "error": f"STOP: {tool} has already been called {_repeats[k] - 1} times "
                     "with exactly these arguments this turn.",
            "instruction": ("The answer will not change. Use the value you already "
                            "received. If it looks wrong, say so in your answer and "
                            "explain the discrepancy rather than calling again."),
        })
    return None

def _safe(obj: Any) -> Any:
    """Serialise engine dataclasses without asdict(), which chokes on mappingproxy."""
    import dataclasses as dc
    if dc.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _safe(getattr(obj, f.name)) for f in dc.fields(obj)}
    from collections.abc import Mapping as _M
    if isinstance(obj, _M):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
