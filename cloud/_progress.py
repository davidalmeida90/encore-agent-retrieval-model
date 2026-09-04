"""Read vLLM's own progress lines, so the launcher can draw a real bar.

The two slow startup phases both report themselves, so there is nothing to
invent and nothing to estimate. Downloading 31 GB and loading it into VRAM are
the parts worth watching; everything else is over in seconds.

    Loading safetensors checkpoint shards:  48% Completed | 32/66
    Fetching 81 files:  45%|####      | 36/81

A timer pretending to be a progress bar is worse than no bar: it keeps moving
when the thing behind it has stalled. These come from the server.
"""

from __future__ import annotations

import re

# "Loading safetensors checkpoint shards:  48% Completed | 32/66"
_SHARDS = re.compile(
    r"Loading safetensors checkpoint shards:\s*(\d+)%[^|]*\|\s*(\d+)/(\d+)"
)

# huggingface_hub's multi-file downloader: "Fetching 81 files:  45%|"
_FETCH = re.compile(r"Fetching\s+(\d+)\s+files[^%]*?(\d{1,3})%")

# Any other tqdm bar, as a fallback while a single large file streams.
_TQDM = re.compile(r"(\d{1,3})%\|")


def progress(log: str, phase: str) -> tuple[float | None, str]:
    """Fraction complete (0-1) and a short label, or (None, "") if unknown.

    The LAST match wins rather than the first. tqdm rewrites its line in place
    on a terminal, but a redirected log keeps every revision, so the end of the
    file is the present moment.
    """
    if phase == "load":
        found = _SHARDS.findall(log)
        if found:
            percent, done, total = found[-1]
            return int(percent) / 100.0, f"{done} of {total} shards"
        return None, ""

    if phase == "download":
        found = _FETCH.findall(log)
        if found:
            files, percent = found[-1]
            return int(percent) / 100.0, f"{files} files"
        bars = _TQDM.findall(log)
        if bars:
            return int(bars[-1]) / 100.0, "31 GB"
    return None, ""


def bar(value: float, width: int = 26) -> str:
    """A rich-markup bar, in ASCII.

    Block-drawing characters look better and crash the legacy Windows console:
    rich falls back to a cp1252 renderer there and raises UnicodeEncodeError on
    the first shaded square. A launcher that dies while reporting progress is
    worse than an ugly bar.
    """
    filled = max(0, min(width, round(value * width)))
    return ("[green]" + "=" * filled + "[/]"
            + "[dim]" + "-" * (width - filled) + "[/]")
