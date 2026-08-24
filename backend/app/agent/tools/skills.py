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

_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


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
