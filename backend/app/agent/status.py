"""Map internal agent/tool events to pipeline status.

TWO audiences, so two events per tool call:

* a friendly stage ("Searching SEC filings…") for the line under the composer,
  which an analyst reads while waiting;
* a raw ``tool`` event naming the tool and its arguments, for the activity log,
  where the whole point is seeing exactly what ran and in what order.

_tool_start_status used to be the only output, and it collapsed every non-
retrieval tool into "Reading source documents…", so the log could not show that
get_sec_financials, run_dcf_valuation and get_stock_prices were different things.
"""

from __future__ import annotations

from app.agent.deps import DocumentAgentDeps
from app.agent.progress import report_progress


def emit_tool_start(deps: DocumentAgentDeps, name: str, detail: str) -> None:
    report_progress(f"tool {name} start {detail}")
    stage, message = _tool_start_status(name, detail)
    deps.emit_status(stage, message)
    # Raw event for the activity log. The UI hides "tool" from the transient
    # status line, so this adds detail without making the wait noisier.
    deps.emit_status("tool", f"{name}({detail})" if detail else name)


def emit_usage_breakdown(deps: DocumentAgentDeps, result: object) -> None:
    """Emit the token attribution as one status event carrying JSON.

    Sent on the existing status stream rather than through a new endpoint, so the
    UI needs no extra request and the numbers arrive with the answer.
    """
    from app.agent import usage as usage_module
    from app import telemetry

    breakdown = usage_module.build(result, counters=telemetry.current())
    deps.emit_status("usage_detail", breakdown.to_json())


def emit_agent_start(deps: DocumentAgentDeps, *, model: str, request_limit: int) -> None:
    report_progress(
        f"agent run start model={model} request_limit={request_limit}"
    )
    deps.emit_status("analyzing", "Analyzing your question…")


def emit_agent_done(
    deps: DocumentAgentDeps,
    *,
    requests: int,
    tool_calls: int,
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
    report_progress(
        "agent run done "
        f"requests={requests} tool_calls={tool_calls} "
        f"input_tokens={input_tokens} output_tokens={output_tokens}"
    )
    # What the run actually cost, so the log answers "why was that slow/expensive"
    # without digging through server output.
    deps.emit_status(
        "usage",
        f"{requests} model requests · {tool_calls} tool calls · "
        f"{input_tokens or 0:,} in / {output_tokens or 0:,} out tokens",
    )
    deps.emit_status("verifying", "Verifying citations…")


def _tool_start_status(name: str, detail: str) -> tuple[str, str]:
    if name == "search_filings":
        suffix = f" ({detail})" if detail != "no filters" else ""
        return "searching", f"Searching SEC filings…{suffix}"
    if name == "read_surrounding_chunks":
        return "reading", "Reading surrounding context…"
    if name in {"read_chunk", "read_chunks"}:
        return "reading", "Reading source passages…"
    return "reading", "Reading source documents…"
