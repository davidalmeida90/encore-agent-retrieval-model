# Running Encore against your own model

Encore talks to any **OpenAI-compatible** endpoint, so the model behind it can be
Gemini, a GPU you rent by the hour, or Ollama on your own machine. Nothing in
`backend/` knows the difference: pick the model in the picker, point
`LOCAL_LLM_BASE_URL` at a server, and the same tools and the same grounding
check apply.

## The cheapest option first: Ollama, no cloud

```bash
ollama pull qwen3:30b
ollama serve
```

Then in `backend/.env`:

```
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
```

Ollama serves the OpenAI shape at `/v1`, so this needs no code change. It is
slower than a rented H100 and free, which is the right trade for trying things.

## Renting a GPU

`serve_on_runpod.py` deploys a pod running vLLM, waits until it answers, and
prints the URL to paste into `backend/.env`.

```bash
cp cloud/.env.example cloud/.env      # add your RunPod and HF keys
python cloud/serve_on_runpod.py up
python cloud/serve_on_runpod.py status
python cloud/serve_on_runpod.py down  # do not forget this one
```

**A pod bills while it exists, not while it works.** An idle pod costs exactly
what a busy one costs, and `down` terminates rather than stops, because a
stopped pod still holds its disk and still charges. `status` prints the running
total so the number is never a surprise.

No SSH is involved. The pod runs `vllm/vllm-openai`, whose entrypoint is the
server, and RunPod publishes the port at
`https://<pod-id>-8000.proxy.runpod.net`. None of your code runs up there.

## Two flags that decide whether this works at all

**`--tool-call-parser`.** Models are trained to emit tool calls as text in their
own format, and vLLM has to translate that back into the `tool_calls` field an
OpenAI client reads. Qwen3.6 uses XML:

```
<tool_call><function=get_sec_financials><parameter=ticker>AAPL</parameter></function></tool_call>
```

so it needs `qwen3_xml`, **not** `hermes`, which expects JSON. With the wrong
parser vLLM returns that XML as ordinary message content, the client sees no
tool call, and the agent looks like it is refusing to use its tools rather than
like a misconfigured server. Check the model's own `tokenizer_config.json`
chat template before guessing.

**`--max-model-len`.** Qwen3.6 advertises a 262,144-token context. Reserving KV
cache for that consumes the card. Encore's largest measured request was about
12,000 tokens, so 32,768 is generous.

## Thinking mode is a cost decision, not a quality switch

Measured on one pod, asking for a single tool call:

```
thinking off      61 output tokens    1.2s
thinking on    1,833 output tokens    9.6s      identical tool call
```

Decode time is linear in output tokens, so thinking is a 30x multiplier on the
part that costs time. `tuning.LOCAL_THINKING_ENABLED` controls it, and it is off
by default for the same reason the cheapest model is the default: most questions
here are lookups.

That is not the whole story though. See `../VIDEO.md` for what happened to the
round count with thinking off, which is the more interesting number.
