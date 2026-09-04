"""THE vLLM AND RUNPOD CONFIG. Rent a GPU, serve an LLM on it, give it back.

## What to change here
##     GPU               which card RunPod rents      (line ~58)
##     MODEL             which checkpoint it serves   (line ~59)
##     VLLM_ARGS         every vLLM flag, each with its reason
##     CONTAINER_DISK_GB sized from the checkpoint, not from habit
##
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

## Normally you do not run this directly; launch.py calls it. Direct use is for
## `status` and `down`.


    python cloud/serve_on_runpod.py up       # deploy, wait until /v1/models answers
    python cloud/serve_on_runpod.py status   # what is running, and what it has cost
    python cloud/serve_on_runpod.py logs     # vLLM stdout, for when it will not start
    python cloud/serve_on_runpod.py down     # terminate EVERY pod on the account

Needs your own RunPod API key in cloud/.env. See cloud/README.md.

## Why there is no SSH here
The pod runs `vllm/vllm-openai`, whose entrypoint is the server itself, and
RunPod publishes any container port at

    https://<pod-id>-8000.proxy.runpod.net

so the OpenAI-compatible API is reachable over ordinary HTTPS. Nothing of ours
runs on the pod, so nothing has to be shipped there and there is nothing to log
into, and no code to ship up.

## The pod bills while it exists, not while it works
An idle pod costs exactly what a busy one costs, so `down` terminates rather
than stops: a stopped pod still holds its disk and still bills. Run `down` when
you finish, not tomorrow.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

# --------------------------------------------------------------- what to run
# Decode is memory-bandwidth bound, so the number that sets answer speed is
# GB/s, not TFLOPS and not VRAM. The model is 27 GB, so 96 GB of VRAM bought
# headroom that was never used. Measured 46.3 tok/s on the RTX PRO 6000, which
# is 1,792 GB/s / 27 GB at 70% efficiency, so the estimate below is calibrated:
#
#   RTX PRO 6000    96G   1,792 GB/s    46 tok/s   $2.09/hr   (measured)
#   H100 SXM        80G   3,350 GB/s    87 tok/s   $2.69/hr
#   H100 NVL        94G   3,900 GB/s   101 tok/s   $2.59/hr   <- 2.2x for 1.24x
#   H200 SXM       141G   4,800 GB/s   124 tok/s   $3.59/hr
#
# Two RTX PRO 6000s with --tensor-parallel-size 2 also roughly doubles decode,
# but at $4.18/hr it is the worst of these per unit of speed.
# The deploy field wants gpuTypes.id, which is not the displayName shown in
# the console: "H100 NVL" is rejected as an unknown GPU type.
GPU = "NVIDIA H100 NVL"  # 94 GB HBM3, 3.9 TB/s
MODEL = "Qwen/Qwen3.8-27B-FP8"

# A persistent disk, in one region, holding the weights. Without it every launch
# re-downloads ~27 GB before the server can start; with it a restart is a load
# rather than a download. Network volumes are region locked, so the pod must be
# placed in the same datacenter, and that datacenter must actually stock the GPU:
# EU-RO-1 tops out at 32 GB cards, which is why this one lives in Iceland.
# Leave both blank to download fresh each time.
# Region-locked to the datacenter that holds it, and EUR-IS-1 stocks RTX PRO
# 6000 but not H100 NVL, so keeping the volume rules out the faster card.
# Blank means "deploy anywhere and download the weights", which costs a few
# minutes per launch and buys the whole GPU catalogue.
NETWORK_VOLUME_ID = ""  # was "h5dpa1c1fb" in EUR-IS-1
DATA_CENTER = "EUR-IS-1"
VOLUME_MOUNT = "/runpod-volume"
SERVED_AS = "qwen"  # what LOCAL_LLM_BASE_URL clients ask for
POD_NAME = "encore-qwen"

# Container disk holds the unpacked vLLM image plus the weights, and getting it
# wrong fails silently: the download fills the disk, the container dies, RunPod
# restarts it, and the only symptom is a pod whose uptime never passes single
# digits. Indistinguishable from a slow load unless you are reading the log.
#
# Sized from the actual checkpoint rather than from habit. Qwen3.8-27B-FP8 is
# 30.9 GB across 81 files (Hugging Face API, checked 2026-08-30):
#
#     vLLM image, unpacked      ~16 GB
#     weights                    31 GB
#     torch.compile cache         ~2 GB
#     logs and temp               ~1 GB
#     ----------------------------------
#     needed                     ~50 GB
#
# 60 leaves about 20% headroom, which covers a partial file left behind by a
# retried download. 80 was carried over from a bf16 checkpoint at 69 GB and is
# simply wasted here. Raise it before switching to a larger or unquantised model:
# this number must lead the checkpoint, not follow it.
CONTAINER_DISK_GB = 60

# vLLM flags, each one deliberate:
#
#   --tool-call-parser qwen3_xml
#       NOT hermes. Qwen3.6's chat template emits tool calls as XML
#       (<function=name><parameter=x>...) rather than the JSON hermes expects.
#       With the wrong parser vLLM returns that XML as ordinary message content,
#       the client sees tool_calls=None, and the agent looks like it is refusing
#       to use its tools. Verified against the model's own tokenizer_config.
#
#   --max-model-len 32768
#       The model advertises 262,144. Reserving KV cache for a 256k context
#       would consume the card. Encore's largest measured question was ~12k
#       tokens in a single request, so 32k is generous.
#
#   --enable-auto-tool-choice
#       Required before vLLM will parse tool calls at all.
#
#   speculative decoding: NOT POSSIBLE on this model, do not retry
#       Investigated properly on 2026-08-30 and the answer is upstream, not
#       configuration. Two spellings were tried:
#           --speculative-config '{"method":"ngram",...}'   crash, quoting
#           --speculative-config.method ngram ...           parsed correctly
#       The dotted form is accepted: vLLM echoed
#           speculative_config={'method':'ngram','num_speculative_tokens':5,...}
#       and then crash-looped anyway, dying silently at cache setup with no
#       traceback, roughly nine minutes into every start.
#
#       The reason is the architecture. Qwen3.8-27B resolves to
#       Qwen3_5ForConditionalGeneration and is a HYBRID: the log shows
#       mamba_mixer2 and qwen_gdn_attention_core, so most layers are Gated
#       DeltaNet linear attention rather than ordinary attention. Rejecting a
#       draft token means rewinding a recurrent SSM state, which is far harder
#       than dropping a KV entry, and vLLM does not do it. Upstream has it open
#       as vllm#39273, "Ngram speculative decoding produces corrupted output on
#       hybrid GDN (Qwen3.5) models", plus vllm#39809 for the prefix-caching
#       crash on hybrid Mamba.
#
#       So even a build that started would return wrong tokens. Speculation on
#       this model needs upstream support; the lever for a faster answer here is
#       fewer output tokens, or two cards with --tensor-parallel-size 2.
#
#   --max-model-len 65536
#       Was 32768, and one narrative question reached 28,731. Hitting the ceiling
#       is a hard failure, not a slow answer, and the card has ~60 GB of KV
#       headroom, so the old value bought nothing.
#
#   --enable-prefix-caching
#       Instructions plus tool schemas are ~2,700 identical tokens at the front
#       of every round, and an agent turn is many rounds. Caching that prefix
#       skips re-prefilling it each time.
VLLM_ARGS = " ".join(
    [
        f"--model {MODEL}",
        f"--served-model-name {SERVED_AS}",
        "--enable-prefix-caching",
        # CAG needs the whole filing resident. Measured with vLLM's own
        # tokenizer, an NVDA 10-K plus the instructions is 191,131 tokens -- not
        # the ~156k a chars/3.8 estimate suggested, because filings are dense
        # with numbers and table markup and tokenise at about 3.16 chars each.
        # 163,840 refused every CAG question; this is the model's full window.
        #
        # KV cost stays modest because the model is hybrid: only 16 of 64 layers
        # keep a cache, at 64 KB/token, so ~16.8 GB at full length against ~53 GB
        # free on an H100 NVL after the weights.
        "--max-model-len 262144",
        "--enable-auto-tool-choice",
        "--tool-call-parser qwen3_xml",
        "--gpu-memory-utilization 0.90",
        "--port 8000",
    ]
)


# Fields that must never reach a terminal or a transcript. RunPod echoes a pod's
# whole `env` back on its detail endpoints, HF_TOKEN and all, so anything that
# prints an API response verbatim leaks it. Learned the hard way on 2026-08-29.
REDACT = ("HF_TOKEN", "RUNPOD_API_KEY", "env")


def safe_print(payload: object) -> None:
    """Print an API response with secret-bearing fields removed."""
    if isinstance(payload, dict):
        payload = {k: ("<redacted>" if k in REDACT else v) for k, v in payload.items()}
    print(payload)


def gql(query: str, key: str) -> dict:
    """POST a GraphQL query. RunPod 403s without a browser-ish User-Agent."""
    # Key goes in a header, never the query string: a URL ends up in proxy logs,
    # shell history and any error message that echoes the request.
    req = urllib.request.Request(
        API,
        data=json.dumps({"query": query}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": UA,
        },
    )
    try:
        body = json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"RunPod HTTP {exc.code}: {exc.read()[:400].decode(errors='replace')}")
    if "errors" in body:
        raise SystemExit("RunPod error: " + json.dumps(body["errors"])[:500])
    return body["data"]


def _dotenv() -> dict[str, str]:
    """Read cloud/.env into a dict. No dependency, and nothing is ever printed.

    Values already in the real environment win, so `set RUNPOD_API_KEY=...` in a
    shell overrides the file rather than the other way round.
    """
    found: dict[str, str] = {}
    path = Path(__file__).with_name(".env")
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                found[k.strip()] = v.strip().strip('"').strip("'")
    return found


def _need(name: str) -> str:
    value = os.environ.get(name) or _dotenv().get(name, "")
    if not value:
        raise SystemExit(
            f"{name} is not set. "
            "Copy cloud/.env.example to cloud/.env and fill it in. "
            "RunPod key: https://console.runpod.io/user/settings"
        )
    return value


def key() -> str:
    return _need("RUNPOD_API_KEY")


def endpoint(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net/v1"


def up() -> None:
    hf = os.environ.get("HF_TOKEN") or _dotenv().get("HF_TOKEN", "")
    # HF_HOME on the volume is what makes the weights persist: the loader finds
    # the cached snapshot and skips the download, and on a cold volume it fills
    # the cache for next time. Both cases are correct, which is why this needs no
    # check that the model is already there.
    env_pairs = [("HF_TOKEN", hf)]
    placement = ""
    if NETWORK_VOLUME_ID:
        placement = (
            f"\n        networkVolumeId: {json.dumps(NETWORK_VOLUME_ID)}"
            f"\n        dataCenterId: {json.dumps(DATA_CENTER)}"
            f"\n        volumeMountPath: {json.dumps(VOLUME_MOUNT)}"
        )
        env_pairs.append(("HF_HOME", VOLUME_MOUNT))
    env_block = ", ".join(
        "{{ key: {}, value: {} }}".format(json.dumps(k), json.dumps(v))
        for k, v in env_pairs
    )
    # Values are JSON-encoded rather than interpolated raw: the vLLM args
    # contain spaces and the token must not be mangled.
    query = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: ALL
        gpuCount: 1
        volumeInGb: 0
        containerDiskInGb: {CONTAINER_DISK_GB}{placement}
        minVcpuCount: 8
        minMemoryInGb: 32
        gpuTypeId: {json.dumps(GPU)}
        name: {json.dumps(POD_NAME)}
        imageName: "vllm/vllm-openai:latest"
        ports: "8000/http"
        dockerArgs: {json.dumps(VLLM_ARGS)}
        env: [{env_block}]
      }}) {{ id costPerHr machine {{ podHostId }} }}
    }}
    """
    pod = gql(query, key())["podFindAndDeployOnDemand"]
    pod_id = pod["id"]
    print(f"pod {pod_id} deploying, ${pod['costPerHr']}/hr")
    print(f"endpoint will be {endpoint(pod_id)}")
    print("waiting for vLLM (weights download plus load, several minutes)...", flush=True)
    wait_ready(pod_id)


def up_debug() -> None:
    """Deploy with vLLM's output served over a second port, so a container that
    dies at startup can still be read.

    Neither RunPod API exposes container logs: the GraphQL schema has no `logs`
    field on PodRuntimeContainer, and REST answers 400 for /pods/{id}/logs. That
    gap is what turns one crash-looping flag into a series of blind redeploys, so
    the fix is to make the container publish its own log.

    `dockerEntrypoint` replaces the image's entrypoint with a real shell and
    `dockerStartCmd` is the script it runs, which also means quoting behaves
    predictably here in a way it does not through `dockerArgs`.

    Costs the same per hour as `up`. Use it the moment a deploy fails twice,
    rather than guessing a third time.
    """
    hf = os.environ.get("HF_TOKEN") or _dotenv().get("HF_TOKEN", "")
    env = {"HF_TOKEN": hf}
    if NETWORK_VOLUME_ID:
        env["HF_HOME"] = VOLUME_MOUNT
    body = {
        "name": POD_NAME + "-debug",
        "imageName": "vllm/vllm-openai:latest",
        "gpuTypeIds": [GPU],
        "gpuCount": 1,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "volumeInGb": 0,
        "ports": ["8000/http", "8001/http"],
        "env": env,
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [
            "mkdir -p /var/log/vllm && cd /var/log/vllm && "
            "(python3 -m http.server 8001 >/dev/null 2>&1 &) && "
            "vllm serve " + VLLM_ARGS.replace("--model ", "")
            + " > /var/log/vllm/out.log 2>&1"
        ],
    }
    if NETWORK_VOLUME_ID:
        body["networkVolumeId"] = NETWORK_VOLUME_ID
        body["dataCenterIds"] = [DATA_CENTER]
    req = urllib.request.Request(
        REST + "/pods", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA,
                 "Authorization": "Bearer " + key()},
    )
    try:
        pod = json.load(urllib.request.urlopen(req, timeout=90))
    except urllib.error.HTTPError as exc:
        raise SystemExit("REST %s: %s" % (exc.code, exc.read()[:400].decode(errors="replace")))
    pid = pod.get("id")
    print("pod %s deploying (debug)" % pid)
    print("log at  https://%s-8001.proxy.runpod.net/out.log" % pid)
    print("api at  %s" % endpoint(pid))


def wait_ready(pod_id: str, minutes: int = 25) -> None:
    """Poll /v1/models until the server answers, or give up loudly.

    Readiness is the HTTP endpoint, not the pod status: RunPod reports RUNNING
    as soon as the container starts, which is long before 35 GB of weights are
    downloaded and loaded.
    """
    url = endpoint(pod_id) + "/models"
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": UA}), timeout=15
            ) as r:
                served = [m["id"] for m in json.load(r).get("data", [])]
                print(f"\nREADY. serving {served}")
                print(f"LOCAL_LLM_BASE_URL={endpoint(pod_id)}")
                return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(20)
    print(f"\nnot ready after {minutes} min. Run: python cloud/serve_on_runpod.py logs")


def pods() -> dict:
    q = """{ myself { clientBalance pods {
        id name desiredStatus costPerHr runtime { uptimeInSeconds }
        machine { gpuDisplayName } } } }"""
    return gql(q, key())["myself"]


def status() -> None:
    me = pods()
    print(f"balance: ${me.get('clientBalance', 0):.2f}")
    if not me["pods"]:
        print("no pods running")
        return
    for p in me["pods"]:
        up_s = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
        print(
            f"  {p['id']}  {p['name']}  {p['desiredStatus']}  "
            f"${p.get('costPerHr', 0)}/hr  up {up_s // 60}m  "
            f"spent ~${(p.get('costPerHr') or 0) * up_s / 3600:.2f}"
        )
        print(f"    {endpoint(p['id'])}")


def logs() -> None:
    for p in pods()["pods"]:
        print(f"=== {p['id']} ===")
        try:
            data = gql(f'{{ pod(input:{{podId:{json.dumps(p["id"])}}}) '
                       f"{{ runtime {{ container {{ logs }} }} }} }}", key())
            print((data["pod"]["runtime"]["container"]["logs"] or "")[-4000:])
        except SystemExit as exc:
            # Logs are not exposed on every plan; the console always has them.
            print(f"could not fetch logs ({exc}). Use the RunPod web console.")


def down() -> None:
    for p in pods()["pods"]:
        gql(f'mutation {{ podTerminate(input:{{podId:{json.dumps(p["id"])}}}) }}', key())
        print(f"terminated {p['id']}")
    print("billing stopped")


if __name__ == "__main__":
    {"up": up, "up-debug": up_debug, "status": status, "logs": logs, "down": down}[
        sys.argv[1] if len(sys.argv) > 1 else "status"
    ]()
