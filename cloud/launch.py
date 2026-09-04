"""THE SCRIPT YOU RUN. One command: rent a GPU, watch it come up, open the app.

## Where this sits (the three-file tour)
##
##     cloud/launch.py            <- THE ONE YOU RUN. Rents the GPU, shows
##                                   progress, starts the app, opens the browser.
##     cloud/serve_on_runpod.py   <- WHAT runs on that GPU: which card, which
##                                   model, which vLLM flags. Also up/down/status.
##     backend/app/agent/models.py<- WHICH MODELS THE UI OFFERS. A registry entry
##                                   per model; the picker is built from it.
##
## The split matters: serve_on_runpod decides what the SERVER hosts, models.py
## decides what the USER can pick. Both must name the same model for the local
## option to work, and they are joined by LOCAL_LLM_BASE_URL in backend/.env,
## which launch.py rewrites for you.

## How to run it

    python cloud/launch.py          # bring everything up, then open the browser
    python cloud/launch.py --keep   # same, but do not offer to terminate on exit

## Why this exists rather than `serve_on_runpod.py up`
`up` prints a row of dots for eight to twelve minutes. That is not merely
unfriendly: it hides the two failures that actually happen. A pod that
crash-loops looks exactly like a pod that is still loading, and a pod that is
quietly downloading 27 GB looks exactly like one that has hung. Both were
diagnosed the slow way more than once.

So this launcher makes the pod publish its own vLLM log on a second port, reads
it, and shows which phase the server is actually in. The phases are real
milestones parsed out of that log, not a timer pretending to be progress.

## What it shows
    cost      accruing live, because the meter starts at deploy, not at ready
    phase     download -> load weights -> compile -> serve, ticked off as they pass
    log       the last line vLLM printed, so a stall has a visible cause

## What it does at the end
Starts the backend and the frontend, waits for both to answer, and opens the
browser. The GPU is useless on its own; the point is the app.
"""

from __future__ import annotations

import argparse
import atexit
import json
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import serve_on_runpod as pod  # noqa: E402  same directory, shares the API plumbing
from _progress import bar, progress  # noqa: E402  vLLM's own progress lines

from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
PYTHON = BACKEND / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():  # posix layout
    PYTHON = BACKEND / ".venv" / "bin" / "python"

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"


@dataclass
class Phase:
    """One milestone, recognised by any of its markers appearing in the log."""

    key: str
    label: str
    markers: tuple[str, ...]
    done: bool = False
    at: float | None = None


def phases() -> list[Phase]:
    """The real startup sequence, in the order vLLM prints it.

    Markers are substrings of lines this exact image emits. They are checked
    cumulatively: seeing a later phase implies the earlier ones passed, because
    the log for a fast start can skip past a marker between two polls.
    """
    return [
        Phase("boot", "renting the machine", ("non-default args",)),
        Phase("resolve", "reading model config", ("Resolved architecture", "Using max model len")),
        Phase("download", "downloading weights (31 GB)", ("Downloading", "%|", "It/s",
                                                          "model weights take")),
        Phase("load", "loading weights into VRAM", ("Loading safetensors", "Loading weights",
                                                    "Initializing a V1 LLM engine")),
        Phase("compile", "compiling CUDA kernels", ("torch.compile took", "Capturing CUDA graphs",
                                                    "Initial profiling")),
        Phase("serve", "server ready", ("Application startup complete", "Uvicorn running")),
    ]


@dataclass
class State:
    pod_id: str = ""
    cost_per_hr: float = 0.0
    started: float = field(default_factory=time.time)
    log_url: str = ""
    api_url: str = ""
    last_line: str = ""
    raw_log: str = ""
    log_bytes: int = 0
    restarts: int = 0
    steps: list[Phase] = field(default_factory=phases)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def spent(self) -> float:
        return self.cost_per_hr * self.elapsed / 3600.0


def _get(url: str, timeout: float = 12.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": pod.UA})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def render(state: State, note: str = "") -> Group:
    header = Table.grid(padding=(0, 2))
    header.add_column(style="dim", justify="right")
    header.add_column()
    header.add_row("gpu", f"[bold]{pod.GPU}[/]")
    header.add_row("model", pod.MODEL)
    header.add_row("pod", state.pod_id or "[yellow]renting, nothing billing yet[/]")
    header.add_row(
        "cost",
        f"[bold]${state.cost_per_hr:.2f}/hr[/]   "
        f"elapsed [bold]{int(state.elapsed // 60)}m {int(state.elapsed % 60):02d}s[/]   "
        f"spent [bold]${state.spent:.2f}[/]",
    )

    steps = Table.grid(padding=(0, 2))
    steps.add_column(width=6, justify="right")
    steps.add_column(width=4)
    steps.add_column()
    total = len(state.steps)
    current = next((i for i, s in enumerate(state.steps) if not s.done), total)
    for index, step in enumerate(state.steps):
        number = f"step {index + 1}"
        if step.done:
            mark, style = "[green]done[/]", "green"
            suffix = f"  [dim]{step.at:.0f}s[/]" if step.at else ""
        elif index == current:
            mark, style = "[yellow]>>[/]", "bold yellow"
            # The two long phases report their own progress; show it rather
            # than a spinner, so a stall is visible as a bar that stops moving.
            fraction, label = progress(state.raw_log, step.key)
            suffix = (f"  {bar(fraction)} [bold]{fraction:.0%}[/] [dim]{label}[/]"
                      if fraction is not None else "  [dim]working[/]")
        else:
            mark, style = "[dim]--[/]", "dim"
            suffix = ""
        steps.add_row(f"[dim]{number}[/]", mark,
                      Text.from_markup(f"[{style}]{step.label}[/]{suffix}"))

    body = [Panel(header, title="encore", border_style="cyan"), steps]
    if state.restarts:
        body.append(Text.from_markup(
            f"[yellow]container restarted {state.restarts}x - it is crash looping, "
            f"not loading. The log line below is the cause.[/]"))
    if state.last_line:
        body.append(Panel(Text(state.last_line[-300:], overflow="fold"),
                          title="vllm", border_style="dim"))
    if note:
        body.append(Text.from_markup(note))
    return Group(*body)


def advance(state: State, log: str) -> None:
    """Mark phases reached, newest wins, and notice a restart."""
    if len(log) < state.log_bytes:
        # The container truncates the log on restart, so it only ever gets
        # shorter when the process died and came back.
        state.restarts += 1
        for step in state.steps:
            step.done, step.at = False, None
    state.log_bytes = len(log)

    state.raw_log = log
    lines = [line for line in log.splitlines() if line.strip()]
    if lines:
        state.last_line = lines[-1]

    highest = -1
    for index, step in enumerate(state.steps):
        if any(marker in log for marker in step.markers):
            highest = max(highest, index)

    # A phase whose marker has appeared is STARTED, not finished. Marking it
    # done on its own first line was why the progress bars never showed: the
    # shard loader announces itself at 0/66, and the step went green instantly.
    # It is finished only once a later phase has spoken.
    for index, step in enumerate(state.steps):
        if index < highest and not step.done:
            step.done, step.at = True, state.elapsed


def deploy(console: Console, state: State) -> None:
    """Start the pod with its log published, so progress is readable."""
    hf = pod._dotenv().get("HF_TOKEN", "")
    env = {"HF_TOKEN": hf}
    if pod.NETWORK_VOLUME_ID:
        env["HF_HOME"] = pod.VOLUME_MOUNT
    body = {
        "name": pod.POD_NAME,
        "imageName": "vllm/vllm-openai:latest",
        "gpuTypeIds": [pod.GPU],
        "gpuCount": 1,
        "containerDiskInGb": pod.CONTAINER_DISK_GB,
        "volumeInGb": 0,
        "ports": ["8000/http", "8001/http"],
        "env": env,
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [
            "mkdir -p /var/log/vllm && cd /var/log/vllm && "
            "(python3 -m http.server 8001 >/dev/null 2>&1 &) && "
            "vllm serve " + pod.VLLM_ARGS.replace("--model ", "")
            + " > /var/log/vllm/out.log 2>&1"
        ],
    }
    if pod.NETWORK_VOLUME_ID:
        body["networkVolumeId"] = pod.NETWORK_VOLUME_ID
        body["dataCenterIds"] = [pod.DATA_CENTER]

    request = urllib.request.Request(
        pod.REST + "/pods", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": pod.UA,
                 "Authorization": "Bearer " + pod.key()})
    for attempt in range(1, 6):
        try:
            created = json.load(urllib.request.urlopen(request, timeout=90))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode(errors="replace")
            if "SUPPLY" in detail.upper() or exc.code in (429, 503):
                console.print(f"[yellow]no capacity for {pod.GPU}, retry {attempt}/5[/]")
                time.sleep(20)
                continue
            raise SystemExit(f"RunPod {exc.code}: {detail}")
    else:
        raise SystemExit(f"No {pod.GPU} available after 5 attempts. Try another GPU "
                         f"in serve_on_runpod.py, or a datacenter with capacity.")

    state.pod_id = created.get("id", "")
    state.cost_per_hr = float(created.get("costPerHr") or 0.0)
    state.log_url = f"https://{state.pod_id}-8001.proxy.runpod.net/out.log"
    state.api_url = f"https://{state.pod_id}-8000.proxy.runpod.net/v1"


def wait_for_server(console: Console, state: State, minutes: int) -> bool:
    """Follow the pod's own log until the API answers."""
    deadline = time.time() + minutes * 60
    with Live(render(state), console=console, refresh_per_second=4) as live:
        while time.time() < deadline:
            try:
                log = _get(state.log_url, timeout=10)
                advance(state, log)
            except Exception:
                pass  # the log server is not up in the first seconds

            try:
                _get(state.api_url + "/models", timeout=10)
                # The API answering is proof every earlier phase completed,
                # whether or not its marker happened to land between two polls.
                for step in state.steps:
                    step.at = step.at or state.elapsed
                    step.done = True
                live.update(render(state, "[green]server is up[/]"))
                return True
            except Exception:
                pass

            live.update(render(state))
            time.sleep(5)
        live.update(render(state, "[red]timed out. The log panel above says why.[/]"))
    return False


def point_backend_at(state: State) -> None:
    """Rewrite LOCAL_LLM_BASE_URL so the app talks to this pod."""
    env_path = BACKEND / ".env"
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    out, seen = [], False
    for line in lines:
        if line.startswith("LOCAL_LLM_BASE_URL="):
            out.append(f"LOCAL_LLM_BASE_URL={state.api_url}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"LOCAL_LLM_BASE_URL={state.api_url}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _alive(url: str) -> bool:
    try:
        _get(url, timeout=4)
        return True
    except Exception:
        return False


def stop_stale_backend(console: Console) -> None:
    """Kill any uvicorn holding a stale LOCAL_LLM_BASE_URL, so it is restarted."""
    if not _alive(BACKEND_URL + "/health"):
        return
    console.print("[dim]restarting the backend so it picks up the new pod URL[/]")
    if sys.platform == "win32":
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*uvicorn*' } | "
             "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        subprocess.run(["pkill", "-f", "uvicorn app.main:app"], check=False)
    for _ in range(10):
        if not _alive(BACKEND_URL + "/health"):
            return
        time.sleep(1)


def start_app(console: Console) -> bool:
    """Bring up the backend and the frontend, and wait for both to answer."""
    started = _CHILDREN

    # A backend that is ALREADY running is not good enough. Settings are read
    # once at import, so a process started against a previous pod keeps that
    # pod's URL in memory no matter what .env now says. Every local-model
    # question then goes to a machine that no longer exists and comes back as
    # "Search failed", with both servers answering 200 and nothing in the log.
    #
    # Diagnosed the slow way: the backend had started 08/30 16:47 and .env was
    # rewritten 08/31 00:27. So it is stopped and restarted unconditionally.
    stop_stale_backend(console)

    if not _alive(BACKEND_URL + "/health"):
        started.append(subprocess.Popen(
            [str(PYTHON), "-m", "uvicorn", "app.main:app", "--port", "8000"],
            cwd=str(BACKEND), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    if not _alive(FRONTEND_URL):
        started.append(subprocess.Popen(
            ["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev"],
            cwd=str(FRONTEND), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    progress = Progress(SpinnerColumn(), TextColumn("{task.description}"),
                        BarColumn(bar_width=24), console=console)
    with progress:
        task = progress.add_task("starting backend and frontend", total=60)
        for _ in range(60):
            if _alive(BACKEND_URL + "/health") and _alive(FRONTEND_URL):
                progress.update(task, completed=60, description="[green]app is up[/]")
                return True
            progress.advance(task)
            time.sleep(1)
        progress.update(task, description="[red]app did not come up[/]")
    return False


_CHILDREN: list[subprocess.Popen] = []
_SHUTDOWN_DONE = False


def shut_everything_down(console: Console, state: State, reason: str = "") -> None:
    """Terminate the pod and the local servers. Safe to call more than once.

    This is the whole safety story of the launcher. A GPU bills whether or not
    anyone is looking at it, and the failure mode is not dramatic: you close a
    window, go to lunch, and $3.19/hr keeps running. So closing the window is
    wired to mean stop, and the pod outliving the process is now the special
    case that has to be asked for with --keep.
    """
    global _SHUTDOWN_DONE
    if _SHUTDOWN_DONE:
        return
    _SHUTDOWN_DONE = True
    if reason:
        console.print(f"\n[yellow]{reason}[/]")

    for child in _CHILDREN:
        try:
            child.terminate()
        except Exception:
            pass

    try:
        for existing in pod.pods().get("pods") or []:
            pod.gql(
                f'mutation {{ podTerminate(input:{{podId:{json.dumps(existing["id"])}}}) }}',
                pod.key(),
            )
            console.print(f"[green]terminated {existing['id']}, billing stopped[/]")
    except Exception as exc:
        console.print(f"[red]COULD NOT STOP THE POD: {exc}[/]")
        console.print("[red]Run this now:  python cloud/serve_on_runpod.py down[/]")
        return

    if state.cost_per_hr:
        console.print(f"[dim]this session cost about ${state.spent:.2f}[/]")


def arm_shutdown(console: Console, state: State) -> None:
    """Make every way of ending this process stop the pod.

    Ctrl+C and a terminate signal are the easy ones. Closing the console window
    on Windows is not: it sends CTRL_CLOSE_EVENT, and neither atexit nor a
    signal handler reliably runs. SetConsoleCtrlHandler does, and Windows gives
    roughly five seconds before killing the process, which is enough for one
    HTTPS call.
    """
    def handler(*_args):
        shut_everything_down(console, state, "shutting down, stopping the GPU")
        return True

    atexit.register(lambda: shut_everything_down(console, state))
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_a: (handler(), sys.exit(0)))
        except Exception:
            pass

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            prototype = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            global _WIN_HANDLER  # must outlive this function or it is collected
            _WIN_HANDLER = prototype(lambda event: bool(handler()))
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_WIN_HANDLER, True)
        except Exception:
            console.print("[yellow]could not hook the window close button; "
                          "use Ctrl+C or stop-encore.bat[/]")


def watch(console: Console, state: State) -> None:
    """Stay in the foreground showing the running cost until the user leaves."""
    console.print("\n[bold]Leave this window open while you use the app.[/]")
    console.print("[bold]Closing it, or Ctrl+C, stops the GPU and stops billing.[/]\n")
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                live.update(Text.from_markup(
                    f"  [green]running[/]  {state.pod_id}   "
                    f"[bold]${state.cost_per_hr:.2f}/hr[/]   "
                    f"up [bold]{int(state.elapsed // 60)}m {int(state.elapsed % 60):02d}s[/]   "
                    f"spent [bold]${state.spent:.2f}[/]   "
                    f"[dim]{FRONTEND_URL}[/]"))
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shut_everything_down(console, state, "stopping")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rent a GPU and open Encore on it.")
    parser.add_argument("--keep", action="store_true",
                        help="leave the pod running after this window closes; the "
                             "default is to stop it so nothing bills unwatched")
    parser.add_argument("--minutes", type=int, default=25,
                        help="how long to wait for the server before giving up")
    args = parser.parse_args()

    console = Console()
    state = State()

    me = pod.pods()
    console.print(f"[dim]balance before: ${me.get('clientBalance', 0):.2f}[/]")

    # Attach to a pod that is already up rather than renting a second one.
    # Two pods is the expensive mistake this guards against, and it also makes
    # the launcher usable as a monitor: run it again in a visible terminal to
    # watch a machine that some other process started.
    running = [p for p in (me.get("pods") or []) if p.get("desiredStatus") == "RUNNING"]
    if running:
        existing = running[0]
        state.pod_id = existing["id"]
        state.cost_per_hr = float(existing.get("costPerHr") or 0.0)
        state.log_url = f"https://{state.pod_id}-8001.proxy.runpod.net/out.log"
        state.api_url = pod.endpoint(state.pod_id)
        uptime = (existing.get("runtime") or {}).get("uptimeInSeconds") or 0
        state.started = time.time() - uptime
        console.print(f"[yellow]attaching to pod {state.pod_id}, already running "
                      f"{uptime // 60}m. Not renting a second one.[/]")
    else:
        deploy(console, state)
    console.print(f"pod [bold]{state.pod_id}[/] at [bold]${state.cost_per_hr:.2f}/hr[/]  "
                  f"[dim]{state.log_url}[/]\n")

    # Armed BEFORE the wait: a pod that never finishes loading is still
    # billing, so Ctrl+C during those eight minutes must stop it too.
    if not args.keep:
        arm_shutdown(console, state)

    ready = wait_for_server(console, state, args.minutes)
    if not ready:
        shut_everything_down(console, state,
                             "the server never answered; stopping the GPU")
        raise SystemExit(1)

    point_backend_at(state)
    console.print(f"[dim]backend .env now points at {state.api_url}[/]")

    if start_app(console):
        # Opened here, once both servers answer. start_app blocks until
        # then, so this is the moment the app is genuinely usable.
        import webbrowser

        webbrowser.open(FRONTEND_URL)
        console.print(f"\n[green]open:[/] {FRONTEND_URL}")
        console.print("[dim]pick 'Qwen 27B dense (local)' in the composer; "
                      "thinking off.[/]")

    if args.keep:
        # Opt out: the pod outlives this process. For handing a running
        # machine to something else, and the only path that can leave money
        # burning after the window is gone.
        console.print(f"\n[bold]${state.cost_per_hr:.2f}/hr will KEEP running after this window closes.[/]")
        console.print("Stop it with:  python cloud/serve_on_runpod.py down")
        return

    watch(console, state)


if __name__ == "__main__":
    main()
