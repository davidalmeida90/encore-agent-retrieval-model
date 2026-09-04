"""## What the model is, and what the GPU actually did

## Why this exists
The Tokens panel answers "what did that cost" in tokens. It cannot answer why a
question took twenty seconds, because tokens are not time. Time is set by two
different things, and they behave in opposite ways:

    prefill   the whole prompt at once, compute bound, very efficient per token
    decode    one token at a time, bandwidth bound, ~100x more expensive each

That is why an answer with 48,683 input tokens and 1,123 output tokens spends
most of its GPU time on the 1,123. A panel that shows only token counts hides
the thing that matters.

## Where the numbers come from
Everything under `serving` and `totals` is scraped from the vLLM server's own
Prometheus endpoint, so it is measured rather than modelled. Everything under
`derived` is arithmetic on top of those measurements plus the model's active
parameter count, and is labelled as such.

Nothing here works for hosted providers: Gemini publishes no metrics endpoint,
so the panel says so rather than inventing figures.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.agent.models import choice as model_choice
from app.auth.dependencies import CurrentUser, get_current_user
from app.config import settings

router = APIRouter(prefix="/runtime", tags=["runtime"])

# One scrape line: name{labels} value
_METRIC = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-\d.eE+]+)$")

# Bytes per parameter, used to turn "3B active params" into "GB read per token".
_BYTES_PER_PARAM = {"fp8": 1.0, "int8": 1.0, "int4": 0.5, "awq": 0.5, "bf16": 2.0, "fp16": 2.0}


class RuntimeInfo(BaseModel):
    configured: bool
    reachable: bool
    base_url: str | None = None
    detail: str | None = None
    model: dict[str, Any] | None = None
    serving: dict[str, Any] | None = None
    totals: dict[str, Any] | None = None
    derived: dict[str, Any] | None = None


def _scrape(text: str) -> dict[str, float]:
    """Prometheus text to {name: value}, keeping the last engine's value.

    Labels are dropped deliberately. A single-engine server reports one series
    per metric, and carrying the label set would complicate every lookup below
    for information the panel does not show.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC.match(line.strip())
        if match:
            try:
                out[match.group("name")] = float(match.group("value"))
            except ValueError:
                continue
    return out


def _cache_config(text: str) -> dict[str, str]:
    """Pull the labels off vllm:cache_config_info, which carries them as data."""
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            inside = line[line.find("{") + 1 : line.rfind("}")]
            pairs = re.findall(r'(\w+)="([^"]*)"', inside)
            return dict(pairs)
    return {}


def _bytes_per_param(quantization: str) -> float:
    low = quantization.lower()
    for key, value in _BYTES_PER_PARAM.items():
        if key in low:
            return value
    return 2.0  # assume bf16 when the label is unfamiliar


@router.get("", response_model=RuntimeInfo)
async def runtime_info(
    model: str | None = None,
    _: CurrentUser = Depends(get_current_user),
) -> RuntimeInfo:
    """Live model and GPU figures, for whichever model the UI has selected."""
    entry = model_choice(model)

    if entry.provider != "openai_compatible":
        return RuntimeInfo(
            configured=False,
            reachable=False,
            detail=(
                f"{entry.label} is a hosted model. Providers publish no serving "
                "metrics, so there is no GPU to report on. Select a self-hosted "
                "model to see this panel."
            ),
        )

    base = settings.local_llm_base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    # The RunPod proxy 403s a request with no browser-ish User-Agent, and the
    # failure looks exactly like the server being down.
    headers = {"User-Agent": "Mozilla/5.0", "Authorization": "Bearer local"}

    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            models_response = await client.get(f"{base}/models")
            models_response.raise_for_status()
            served = models_response.json().get("data", [{}])[0]
            metrics_text = (await client.get(f"{root}/metrics")).text
    except Exception as exc:
        return RuntimeInfo(
            configured=True,
            reachable=False,
            base_url=base,
            detail=(
                f"No server answered at {base} ({type(exc).__name__}). "
                "Start one, or point LOCAL_LLM_BASE_URL somewhere else."
            ),
        )

    m = _scrape(metrics_text)
    cache = _cache_config(metrics_text)

    prompt_tokens = m.get("vllm:prompt_tokens_total", 0.0)
    gen_tokens = m.get("vllm:generation_tokens_total", 0.0)
    queries = m.get("vllm:prefix_cache_queries_total", 0.0)
    hits = m.get("vllm:prefix_cache_hits_total", 0.0)
    ttft_sum = m.get("vllm:time_to_first_token_seconds_sum", 0.0)
    ttft_n = m.get("vllm:time_to_first_token_seconds_count", 0.0)
    tpot_sum = m.get("vllm:request_time_per_output_token_seconds_sum", 0.0)
    tpot_n = m.get("vllm:request_time_per_output_token_seconds_count", 0.0)

    avg_tpot = (tpot_sum / tpot_n) if tpot_n else 0.0
    spec = entry.spec

    derived: dict[str, Any] = {}
    if spec:
        # Per generated token the GPU must read every ACTIVE parameter out of
        # memory and do about 2 FLOPs on each. That ratio, ~2 FLOPs per byte, is
        # why decode is bandwidth bound: a modern card needs a few hundred FLOPs
        # per byte before its compute is the limit.
        bytes_per_token = spec.active_params_b * 1e9 * _bytes_per_param(spec.quantization)
        flops_per_token = 2 * spec.active_params_b * 1e9
        derived = {
            "gb_read_per_output_token": round(bytes_per_token / 1e9, 2),
            "gflops_per_output_token": round(flops_per_token / 1e9, 1),
            "arithmetic_intensity_flops_per_byte": round(flops_per_token / bytes_per_token, 1),
            "total_tflops": round(flops_per_token * gen_tokens / 1e12, 1),
            "total_gb_read": round(bytes_per_token * gen_tokens / 1e9, 1),
            # Measured, not modelled: vLLM times prefill and decode separately.
            "measured_prefill_seconds": round(ttft_sum, 1),
            "measured_decode_seconds": round(avg_tpot * gen_tokens, 1),
            "note": (
                "Decode reads the active experts once per token, so its time is "
                "linear in output tokens and set by memory bandwidth. Prefill "
                "processes the prompt in one batch and is set by compute."
            ),
        }

    return RuntimeInfo(
        configured=True,
        reachable=True,
        base_url=base,
        model={
            "label": entry.label,
            "served_as": served.get("id"),
            "checkpoint": served.get("root"),
            "context_window": served.get("max_model_len"),
            "quantization": spec.quantization if spec else None,
            "architecture": spec.architecture if spec else None,
            "modality": spec.modality if spec else None,
            "total_params_b": spec.total_params_b if spec else None,
            "active_params_b": spec.active_params_b if spec else None,
            "experts": spec.experts if spec else None,
            "active_experts": spec.active_experts if spec else None,
        },
        serving={
            "prefix_caching": cache.get("enable_prefix_caching") == "True",
            "prefix_cache_hit_rate": round(hits / queries, 3) if queries else None,
            "kv_cache_tokens": int(float(cache.get("kv_cache_size_tokens", 0) or 0)),
            "kv_cache_dtype": cache.get("cache_dtype"),
            "gpu_memory_utilization": cache.get("gpu_memory_utilization"),
            "requests_running": int(m.get("vllm:num_requests_running", 0)),
        },
        totals={
            "requests": int(ttft_n),
            "prompt_tokens": int(prompt_tokens),
            "generation_tokens": int(gen_tokens),
            "avg_prefill_seconds": round(ttft_sum / ttft_n, 3) if ttft_n else None,
            "decode_tokens_per_second": round(1 / avg_tpot, 1) if avg_tpot else None,
        },
        derived=derived or None,
    )
