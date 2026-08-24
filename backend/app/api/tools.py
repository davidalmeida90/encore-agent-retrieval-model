"""## Expose the tool catalogue to the UI

The Tools panel in the chat header lists what the agent can actually call, and
lets you open each one's source. That source is read from the .py file at request
time rather than stored anywhere, so it cannot drift out of date: what the panel
shows is what the model is running.

Why this is worth having: the schema the model sees is derived from these exact
signatures and docstrings. When a tool misbehaves, the first question is always
"what was it actually told this tool does", and until now the only way to answer
that was to open the repo.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.agent.toolgate import _VALUATION_TOOLS
from app.agent.tools import TOOLS
from app.auth.dependencies import CurrentUser, get_current_user

router = APIRouter(prefix="/tools", tags=["tools"])

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Same grouping the tools package uses, by the question each answers.
_GROUPS = {
    "retrieval": "What management said",
    "fundamentals": "What the numbers are",
    "market": "What it is worth today",
    "valuation": "What it should be worth",
    "skills": "Methodology",
}


class ToolInfo(BaseModel):
    name: str
    group: str
    group_label: str
    description: str
    parameters: list[str]
    source: str
    source_path: str
    line: int | None = None
    valuation_gated: bool = False


def _describe(name: str, fn: Any) -> ToolInfo:
    module = inspect.getmodule(fn)
    module_name = (module.__name__.rsplit(".", 1)[-1] if module else "") or "other"
    try:
        source = inspect.getsource(fn)
        _, line = inspect.getsourcelines(fn)
    except (OSError, TypeError):
        source, line = "# source unavailable", None

    path = ""
    module_file = getattr(module, "__file__", None) if module else None
    if module_file:
        try:
            path = str(Path(module_file).resolve().relative_to(_BACKEND_ROOT))
        except ValueError:
            path = Path(module_file).name

    doc = inspect.getdoc(fn) or ""
    try:
        params = [
            p for p in inspect.signature(fn).parameters if p not in {"ctx", "self"}
        ]
    except (TypeError, ValueError):
        params = []

    return ToolInfo(
        name=name,
        group=module_name,
        group_label=_GROUPS.get(module_name, "Other"),
        description=doc,
        parameters=params,
        source=source,
        source_path=path.replace("\\", "/"),
        line=line,
        # These are hidden from the model unless the question sounds like
        # valuation, because their schemas are half the per-request token budget.
        valuation_gated=name in _VALUATION_TOOLS,
    )


def _catalogue() -> list[ToolInfo]:
    # Read the registry directly rather than the built agent. The agent wraps
    # TOOLS in a FilteredToolset (valuation tools are hidden unless the question
    # calls for them), so asking the agent would return whatever the last request
    # happened to be allowed to see. The registry is the honest full list.
    infos = [_describe(fn.__name__, fn) for fn in TOOLS]
    order = list(_GROUPS)
    infos.sort(key=lambda i: (order.index(i.group) if i.group in order else 99, i.name))
    return infos


@router.get("", response_model=list[ToolInfo])
async def list_tools(_: CurrentUser = Depends(get_current_user)) -> list[ToolInfo]:
    """Every tool registered on the agent, with its source."""
    return _catalogue()


@router.get("/{name}", response_model=ToolInfo)
async def get_tool(
    name: str,
    _: CurrentUser = Depends(get_current_user),
) -> ToolInfo:
    for info in _catalogue():
        if info.name == name:
            return info
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"No tool named {name!r}")
