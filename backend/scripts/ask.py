"""Ask one question and show everything worth seeing, for the least quota.

    python -u scripts/ask.py "What was Apple's capital expenditure in fiscal 2025?"

Prints the answer exactly as the frontend receives it (so Markdown formatting can
be judged), the tool calls in order, the request count, and whatever the daily
quota budget reported. One question, one run, no eval harness around it.
"""

from __future__ import annotations

import sys
import uuid

import structlog

from app import telemetry
from app.agent.agent import get_document_agent
from app.agent.deps import DocumentAgentDeps, TurnRegistry
from app.config import settings
from app.retrieval.retriever import DocumentRetriever
from pydantic_ai import UsageLimits, capture_run_messages

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))

QUESTION = " ".join(sys.argv[1:]) or "What was Apple's capital expenditure in fiscal 2025?"


def main() -> None:
    deps = DocumentAgentDeps(
        retriever=DocumentRetriever(),
        registry=TurnRegistry(),
        user_id=uuid.uuid4(),
        thread_id=uuid.uuid4(),
        on_status=lambda *a, **k: None,
    )
    print(f"model: {settings.gemini_chat_model}")
    print(f"Q: {QUESTION}\n")

    # capture_run_messages keeps the transcript even when the run raises, which is
    # the only way to see WHY a budget tripped: a legitimately big question and a
    # model stuck in a loop both just look like "too many tokens" from outside.
    with capture_run_messages() as messages:
        try:
            result = get_document_agent().run_sync(
                QUESTION,
                deps=deps,
                usage_limits=UsageLimits(request_limit=settings.agent_request_limit),
            )
        except Exception as exc:
            print(f"RUN FAILED: {type(exc).__name__}: {exc}\n")
            calls = [
                (p.tool_name, str(p.args)[:90])
                for m in messages
                for p in m.parts
                if type(p).__name__ == "ToolCallPart"
            ]
            print(f"tool calls made: {len(calls)}")
            for name, args in calls:
                print(f"  {name:22} {args}")
            return

    calls = [
        p.tool_name
        for m in result.all_messages()
        for p in m.parts
        if type(p).__name__ == "ToolCallPart" and getattr(p, "tool_name", "") != "final_result"
    ]
    usage = result.usage

    print("--- ANSWER (as the frontend renders it) " + "-" * 30)
    print(result.output.answer)
    print("-" * 70)
    print(f"citations : {len(result.output.citations)}")
    print(f"tools     : {calls}")
    print(f"requests  : {usage.requests}   tokens in/out: {usage.input_tokens}/{usage.output_tokens}")
    print(f"daily     : {telemetry.daily_usage()}")


if __name__ == "__main__":
    main()
