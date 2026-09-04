"""WHERE THE MODEL PICKER COMES FROM. One entry per model the UI offers.

## The three-file tour
##     cloud/launch.py             the script you RUN
##     cloud/serve_on_runpod.py    WHICH model the GPU serves (MODEL constant)
##     THIS FILE                   WHICH models the user can PICK
##
## An entry here with provider="openai_compatible" is a self-hosted model: it
## points at whatever vLLM is serving on LOCAL_LLM_BASE_URL. So adding a local
## model is two edits, not one - the checkpoint in serve_on_runpod.py, and the
## entry here that lets someone choose it. The dropdown itself is
## frontend/src/components/chat/ModelSelect.tsx and needs no editing; it is
## built from this list over /chat/models.
## The models a question can be answered with

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

## Two providers, one registry
`provider` decides which wire format the model speaks, and nothing else changes.
Tools are declared as plain Python either way: PydanticAI translates the same
`TOOLS` list into Gemini's `function_declarations` or OpenAI's `tools` array, so
adding a self-hosted model costs a registry entry, not a rewrite.

    gemini             Google's API, keyed by GEMINI_API_KEY
    openai_compatible  anything serving OpenAI's HTTP shape: vLLM, llama.cpp,
                       Ollama, LM Studio, or OpenAI itself

## Adding one
Append to MODELS. Any Gemini model listed under generateContent works. For a
self-hosted one, set provider="openai_compatible" and point
LOCAL_LLM_BASE_URL at the server. The agent is rebuilt per model and cached, so
switching costs one construction and nothing after that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provider = Literal["gemini", "openai_compatible"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What the model IS, for the Runtime panel.

    Hosted providers publish none of this, so it is only filled in for models
    that are self-hosted, where you chose the checkpoint and therefore know.
    `active_params_b` is the number that governs speed: decode reads only the
    active experts per token, so it sets both the bandwidth cost per token and
    the arithmetic the panel derives from it.
    """

    total_params_b: float
    active_params_b: float
    quantization: str
    architecture: str
    modality: str
    experts: int | None = None
    active_experts: int | None = None


@dataclass(frozen=True, slots=True)
class ModelChoice:
    id: str
    label: str
    hint: str
    default: bool = False
    provider: Provider = "gemini"
    # What the SERVER calls the model, when that differs from the id shown in
    # the UI. vLLM answers to whatever --served-model-name it was started with.
    served_name: str | None = None
    spec: ModelSpec | None = None


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
    # Both local entries point at the SAME endpoint, because a vLLM server runs
    # one model. Pick the one you actually started; the Runtime panel reads the
    # served checkpoint from the server, so it always shows the truth.
    ModelChoice(
        id="qwen3.8-27b",
        label="Qwen 27B dense (local)",
        hint=(
            "Self-hosted, 27B dense so every parameter fires on every token. "
            "Matched Gemini on round count where the MoE models wandered: 2 "
            "rounds against 5 to 12 on the same question, and the only local "
            "model to finish a DCF."
        ),
        provider="openai_compatible",
        served_name="qwen",
        spec=ModelSpec(
            total_params_b=27.0,
            # Dense, so active equals total. This is the number that predicted
            # agent-loop competence across everything tested: the models that
            # wandered all had ~3B active, whatever their total.
            active_params_b=27.0,
            quantization="FP8 (E4M3, dynamic)",
            architecture="Dense, 64 layers",
            modality="text only",
        ),
    ),
    ModelChoice(
        id="qwen3-30b-a3b",
        label="Qwen 30B (local)",
        hint="Self-hosted MoE, text only, a generation older than the 35B and much better behaved in this loop: 4 rounds against 12 on the same question.",
        provider="openai_compatible",
        served_name="qwen",
        spec=ModelSpec(
            total_params_b=30.5,
            active_params_b=3.3,
            quantization="FP8 (E4M3, dynamic)",
            architecture="Mixture of Experts, 48 layers",
            modality="text only",
            experts=128,
            active_experts=8,
        ),
    ),
    ModelChoice(
        id="qwen3.6-35b-a3b",
        label="Qwen 35B (local)",
        hint="Self-hosted mixture of experts, 35B total and 3B active. No quota and no per-token cost; needs a vLLM server reachable at LOCAL_LLM_BASE_URL.",
        provider="openai_compatible",
        served_name="qwen",
        spec=ModelSpec(
            total_params_b=35.0,
            # 8 of 256 experts fire per token. This is the number that sets
            # decode speed, not the 35: at FP8 it means ~3 GB read per token.
            active_params_b=3.0,
            quantization="FP8 (E4M3, dynamic)",
            architecture="Mixture of Experts, 40 layers",
            modality="text + vision",
            experts=256,
            active_experts=8,
        ),
    ),
]

_BY_ID = {m.id: m for m in MODELS}
DEFAULT_MODEL = next(m.id for m in MODELS if m.default)


def choice(model_id: str | None) -> ModelChoice:
    """The full entry for an id, falling back to the default like resolve()."""
    return _BY_ID[resolve(model_id)]


def resolve(model_id: str | None) -> str:
    """Return a known model id, falling back to the default.

    Unknown ids fall back rather than raising: a stale value in a browser tab
    should not make the chat unusable.
    """
    if model_id and model_id in _BY_ID:
        return model_id
    return DEFAULT_MODEL
