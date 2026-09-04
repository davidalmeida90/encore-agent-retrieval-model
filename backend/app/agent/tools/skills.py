"""Progressive-disclosure skills, using the vendored SkillsLoader unmodified.

Only one-line descriptions ride in the system prompt; a body is pulled on demand.
Mechanism and loader are HKUDS/Vibe-Trading's (vendor/skills.py, MIT); this file
is only the PydanticAI tool wrapper and the description injector.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_ai import RunContext

from app.agent.deps import DocumentAgentDeps
from app.agent.status import emit_tool_start
from vendor.skills import SkillsLoader

# backend/skills, not backend/app/skills. One .parent short of the repo layout
# meant the loader pointed at a directory that has never existed, so
# get_descriptions() returned "(no skills)" and load_skill could only fail. The
# valuation skill is what supplies beta, ERP, terminal growth and the
# reinvestment rule, so every DCF ran on whatever the model invented instead.
_SKILLS_DIR = Path(__file__).resolve().parents[3] / "skills"


@lru_cache(maxsize=1)
def loader() -> SkillsLoader:
    return SkillsLoader(skills_dir=_SKILLS_DIR)


def skill_descriptions() -> str:
    """One line per skill, for the system prompt. The only always-on cost."""
    try:
        return loader().get_descriptions()
    except Exception:
        return ""


# Which skills have been read this turn, keyed by thread. run_dcf_valuation
# consults this: the valuation skill is what supplies beta, ERP, terminal growth
# and the reinvestment rule, and without it the model invents them.
_loaded: dict[str, set[str]] = {}


#: Turns in which `reason_about_assumptions` has run, keyed by thread. Same
#: mechanism as `_loaded`, and for the same reason: a step the answer depends on
#: cannot be left to whether the model felt like following prose.
_reasoned: dict[str, set[str]] = {}


def mark_reasoned(ctx, name: str) -> None:
    _reasoned.setdefault(str(getattr(ctx.deps, "thread_id", "?")), set()).add(name)


def has_reasoned(ctx, name: str) -> bool:
    return name in _reasoned.get(str(getattr(ctx.deps, "thread_id", "?")), set())


def skill_was_loaded(ctx, name: str) -> bool:
    return name in _loaded.get(str(getattr(ctx.deps, "thread_id", "?")), set())


async def load_skill(ctx: RunContext[DocumentAgentDeps], name: str) -> str:
    """Load a skill's methodology before doing the work it covers.

    Call this FIRST when a task matches a skill listed in your instructions.
    A skill defines how to source and justify inputs; skipping it means guessing.
    """
    emit_tool_start(ctx.deps, "load_skill", name)
    _loaded.setdefault(str(getattr(ctx.deps, "thread_id", "?")), set()).add(name)
    return loader().get_content(name)
