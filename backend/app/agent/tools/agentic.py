"""## Retrieval with no index: search, parse a page, prompt a sub-model about it

## Provenance
This is a port of the retrieval design in Vals AI's `finance-agent-v2`, the agent
behind their Finance Agent benchmark, kept deliberately close to the original so
that running it against Encore's questions measures *their method* rather than an
improvement on it. Four tools, in their shape:

    web_search            Tavily, the public internet
    edgar_search          SEC full-text search, metadata only
    parse_html_page       fetch a URL, strip it to text, SAVE IT under a key
    retrieve_information  apply an LLM prompt to saved documents by key

## The idea worth stealing
Documents never enter the conversation. `parse_html_page` puts the text in a
store and returns only a key; `retrieve_information` takes a prompt containing
`{{key}}` placeholders, substitutes the document text at send time, and passes it
to a *second* model. So a 500,000 character 10-K is read without ever occupying
the agent's context window, and a prompt naming two keys compares two filings in
a single call.

## Where this port differs, and why
- **The sub-model is metered.** Every `retrieve_information` call is a billed
  model call that never appears in the agent run's usage, which made index-free
  retrieval look far cheaper than it is. Counted on deps, reported with it.
- **Text that reaches the model is registered as a citable passage.** Their agent
  cites source URLs in prose. Encore's grounding gate verifies every excerpt
  against a passage actually retrieved this turn, so unregistered text leaves the
  model no way to cite except inventing an id, which it duly does.
- **The date ceiling is today, not a constant.** Theirs pins every search to
  2026-03-01 so the benchmark stays reproducible. A live assistant doing that
  would silently ignore every filing since.
- Tavily is called over its REST endpoint with httpx rather than through
  `tavily-python`, to avoid adding a dependency. Same request, same response.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import uuid as _uuid
from datetime import date
from typing import Annotated, Any
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from app.agent.deps import DocumentAgentDeps
from app.agent.status import emit_tool_start
from app.agent.tools._guards import _record_failure, _too_many_repeats, remember
from app.config import settings
from app.retrieval.types import RetrievedPassage

_SEC_API = "https://api.sec-api.io/full-text-search"
_TAVILY_API = "https://api.tavily.com/search"
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_PLACEHOLDER = re.compile(r"{{([^{}]+)}}")

# parse_html_page fetches a URL the MODEL chose, and the model's context is full
# of text this system does not control: filing bodies, and whatever the web
# search returned. A sentence inside either can ask it to read
# http://169.254.169.254/latest/meta-data/, and a server-side fetch of that
# returns cloud credentials. Redirects are followed manually for the same
# reason: a public host is free to answer 302 and point at a private address.
_MAX_REDIRECTS = 5


def _safe_address(url: str) -> tuple[str | None, str | None]:
    """Return (address, None) when this URL may be fetched, else (None, reason).

    The address comes back because validating a hostname and then letting the
    HTTP client resolve it again is a race, not a check: an attacker-controlled
    domain with a one-second TTL can answer with a public address for the check
    and 169.254.169.254 for the connection. That is DNS rebinding, and the only
    cure is to connect to the address that was actually vetted.

    Every address the name resolves to must be public, not just the first: a
    name can answer with one of each and win by whichever gets picked.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None, f"scheme {parsed.scheme or 'missing'!r} is not allowed; use http or https"
    host = parsed.hostname
    if not host:
        return None, "no hostname in the URL"
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return None, f"could not resolve {host!r}"
    chosen: str | None = None
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast
                or address.is_unspecified):
            return None, (f"{host!r} resolves to {address}, which is not a public "
                          f"address. Fetching it would reach this machine or its "
                          f"private network.")
        chosen = chosen or str(address)
    return chosen, None


# Citation ids are derived rather than random, so the same slice of the same
# document is always the same id across runs.
_NS = _uuid.uuid5(_uuid.NAMESPACE_URL, "encore/agentic-retrieval")


def _max_end_date() -> str:
    """Latest date any search may reach."""
    return date.today().isoformat()


def _check_date(field: str, value: str) -> None:
    if not _DATE.match(value):
        raise ValueError(f"Invalid {field} format: {value!r}. Expected YYYY-MM-DD.")


def _headers() -> dict[str, str]:
    """SEC rate limits hard without a contact address in the User-Agent."""
    ua = settings.sec_edgar_ua or "Encore research contact@example.com"
    return {"User-Agent": ua}


def _store(ctx: RunContext[DocumentAgentDeps]) -> dict[str, str]:
    """The agent's data storage: key -> document text, for this turn."""
    existing: dict[str, str] = getattr(ctx.deps, "fetched_documents", None) or {}
    try:
        ctx.deps.fetched_documents = existing  # type: ignore[attr-defined]
    except Exception:
        pass
    return existing


def _register(ctx: RunContext[DocumentAgentDeps], key: str, start: int,
              text: str) -> str | None:
    """Register document text as a citable passage, returning its id.

    Their agent cites source URLs in prose and has no notion of a verifiable
    passage. Encore's gate checks every excerpt against something actually
    retrieved this turn, so text that reaches the model unregistered leaves
    fabrication as the only way to cite it.
    """
    urls: dict[str, str] = getattr(ctx.deps, "document_urls", None) or {}
    try:
        passage_id = _uuid.uuid5(_NS, f"{key}:{start}:{len(text)}")
        ctx.deps.registry.register(RetrievedPassage(
            chunk_id=passage_id,
            document_id=_uuid.uuid5(_NS, key),
            chunk_index=start,
            text=text,
            page=None,
            section=None,
            fusion_score=0.0,
            ticker="",
            company_name=key,
            form="",
            filing_date=date.today(),
            fiscal_year=None,
            accession_number=urls.get(key, key),
        ))
        return str(passage_id)
    except Exception:  # never fail a good tool result over bookkeeping
        return None


async def web_search(
    ctx: RunContext[DocumentAgentDeps],
    search_query: Annotated[str, Field(description="The query to search for")],
    start_date: Annotated[str, Field(description=(
        "(optional) Start of the search range, YYYY-MM-DD. Must not equal end_date."
    ))] = "",
    end_date: Annotated[str, Field(description=(
        "(optional) End of the search range, YYYY-MM-DD. Capped at today."
    ))] = "",
    number_of_results: Annotated[int, Field(ge=1, le=20, description=(
        "(optional) The number of search results to return."
    ))] = 10,
) -> str:
    """Search the public internet. Each result has a url, a title and an excerpt.

    Use this for anything not in an SEC filing. Filings stay authoritative for
    financial figures: where a number appears in both, the filing wins.
    """
    args = {"q": search_query, "s": start_date, "e": end_date, "n": number_of_results}
    if (stop := _too_many_repeats(ctx, "web_search", args)):
        return stop
    emit_tool_start(ctx.deps, "web_search", search_query[:48])

    if not settings.tavily_api_key:
        return json.dumps({"error": "TAVILY_API_KEY is not configured."})

    ceiling = _max_end_date()
    payload: dict[str, Any] = {
        "query": search_query,
        "search_depth": "basic",
        "max_results": number_of_results,
        "chunks_per_source": 1,
    }
    try:
        if end_date:
            _check_date("end_date", end_date)
        payload["end_date"] = min(end_date or ceiling, ceiling)
        if start_date:
            _check_date("start_date", start_date)
            start = min(start_date, ceiling)
            if start > payload["end_date"]:
                raise ValueError(
                    f"start_date {start!r} is later than end_date "
                    f"{payload['end_date']!r}"
                )
            payload["start_date"] = start
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _TAVILY_API,
                json=payload,
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
    except Exception as exc:
        _record_failure(ctx, "web_search")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return remember(ctx, "web_search", args, json.dumps(results, default=str))


async def edgar_search(
    ctx: RunContext[DocumentAgentDeps],
    search_query: Annotated[str, Field(description=(
        "Case-insensitive term or phrase to find in the contents of filings and "
        "their attachments. Supports wildcards (*), OR, NOT, and exact phrases in "
        'quotation marks ("exact phrase"). Terms are joined by an implicit AND.'
    ))],
    form_types: Annotated[list[str], Field(description=(
        "(optional) Limit to specific EDGAR form types, e.g. ['10-K', '10-Q']. "
        "Default: all form types."
    ))] = [],
    ciks: Annotated[list[str], Field(description=(
        '(optional) Limit to specific CIKs, e.g. ["0000320193"]. Leading zeros '
        "optional. Default: all CIKs."
    ))] = [],
    start_date: Annotated[str, Field(description=(
        "(optional) Start of the search range, YYYY-MM-DD."
    ))] = "1900-01-01",
    end_date: Annotated[str, Field(description=(
        "(optional) End of the search range, YYYY-MM-DD. Capped at today."
    ))] = "",
    page: Annotated[int, Field(ge=1, description=(
        "(optional) Each page holds up to 100 filings. Increase for the next 100."
    ))] = 1,
    top_n_results: Annotated[int, Field(ge=1, le=100, description=(
        "(optional) Return only the first N results from the page."
    ))] = 100,
) -> str:
    """Search EDGAR full text through the SEC API.

    Returns filing **metadata** only, never the text. Each result carries the
    document URL: pass that to `parse_html_page` to actually read the filing.
    """
    args = {"q": search_query, "f": form_types, "c": ciks,
            "s": start_date, "e": end_date, "p": page, "n": top_n_results}
    if (stop := _too_many_repeats(ctx, "edgar_search", args)):
        return stop
    emit_tool_start(ctx.deps, "edgar_search", f"{search_query[:40]} {form_types}")

    if not settings.sec_api_key:
        return json.dumps({"error": "SEC_API_KEY is not configured."})

    ceiling = _max_end_date()
    try:
        _check_date("start_date", start_date)
        if end_date:
            _check_date("end_date", end_date)
        start = min(start_date, ceiling)
        end = min(end_date or ceiling, ceiling)
        if start > end:
            raise ValueError(f"start_date {start!r} is later than end_date {end!r}")
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    payload: dict[str, Any] = {"query": search_query, "startDate": start, "endDate": end}
    if page:
        payload["page"] = page
    if form_types:
        payload["formTypes"] = form_types
    if ciks:
        payload["ciks"] = ciks

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _SEC_API,
                json=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": settings.sec_api_key},
            )
            response.raise_for_status()
            filings = response.json().get("filings", [])[:top_n_results]
    except Exception as exc:
        _record_failure(ctx, "edgar_search")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    return remember(ctx, "edgar_search", args, json.dumps(filings, default=str))


async def parse_html_page(
    ctx: RunContext[DocumentAgentDeps],
    url: Annotated[str, Field(description="The URL of the HTML page to parse")],
    key: Annotated[str, Field(description=(
        "The key to save the result under in the conversation's data storage."
    ))],
) -> str:
    """Fetch a page, strip it to plain text, and save it to data storage.

    Returns a receipt, not the document. Filings run to hundreds of thousands of
    characters and would exhaust the context window; use `retrieve_information`
    to ask questions of what was saved.
    """
    args = {"u": url, "k": key}
    if (stop := _too_many_repeats(ctx, "parse_html_page", args)):
        return stop
    emit_tool_start(ctx.deps, "parse_html_page", f"{key} <- {url[:44]}")

    target = url
    try:
        async with httpx.AsyncClient(timeout=60.0, headers=_headers(),
                                         follow_redirects=False) as client:
            for _ in range(_MAX_REDIRECTS):
                address, reason = _safe_address(target)
                if reason or not address:
                    _record_failure(ctx, "parse_html_page")
                    return json.dumps({"error": f"Refused to fetch {target!r}: {reason}"})
                parsed = urlparse(target)
                # Connect to the address that was vetted, carrying the real
                # hostname in the Host header and in the TLS SNI so the request
                # is otherwise unchanged. Handing httpx the hostname would let
                # it resolve again and reopen the rebinding race.
                netloc = address if ":" not in address else f"[{address}]"
                if parsed.port:
                    netloc = f"{netloc}:{parsed.port}"
                pinned = parsed._replace(netloc=netloc).geturl()
                response = await client.get(
                    pinned,
                    headers={"Host": parsed.netloc},
                    extensions={"sni_hostname": parsed.hostname},
                )
                if response.is_redirect and response.headers.get("location"):
                    # Resolve the next hop against the ORIGINAL url, never the
                    # pinned one, then validate it on the next pass.
                    target = urljoin(target, response.headers["location"])
                    continue
                break
            else:
                return json.dumps({"error": f"Too many redirects from {url!r}."})
            response.raise_for_status()
            html = response.text
    except Exception as exc:
        _record_failure(ctx, "parse_html_page")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    if not text:
        return json.dumps({"error": "The page produced no text."})

    store = _store(ctx)
    warning = ("The key already existed in data storage and was overwritten. "
               if key in store else "")
    store[key] = text
    urls: dict[str, str] = getattr(ctx.deps, "document_urls", None) or {}
    urls[key] = url
    try:
        ctx.deps.document_urls = urls  # type: ignore[attr-defined]
    except Exception:
        pass

    result = json.dumps({
        "status": f"{warning}Saved under the key: {key}.",
        "characters": len(text),
        "preview": text[:400],
        "data_storage_keys": sorted(store),
        "next_step": (
            "Call retrieve_information with a prompt containing {{" + key + "}} "
            "to ask about this document."
        ),
    })
    return remember(ctx, "parse_html_page", args, result)


class CharacterRange(BaseModel):
    """A slice of one stored document, so a prompt need not carry all of it."""

    key: Annotated[str, Field(description="The document key from data storage")]
    start: Annotated[int, Field(ge=0, description="Start character index, inclusive")]
    end: Annotated[int, Field(ge=0, description="End character index, exclusive")]


async def retrieve_information(
    ctx: RunContext[DocumentAgentDeps],
    prompt: Annotated[str, Field(description=(
        "The prompt passed to the LLM. It MUST contain at least one data storage "
        "key in the form {{key_name}}, for example "
        "'Summarize this 10-K filing: {{company_10k}}'. The text stored under "
        "each key replaces its placeholder before the prompt is sent. Naming two "
        "keys in one prompt compares two documents in a single call."
    ))],
    input_character_ranges: Annotated[list[CharacterRange], Field(description=(
        "(optional) Pass only part of a document. Each entry gives a key, a "
        "start (inclusive) and an end (exclusive). Any key not listed is "
        "substituted in full."
    ))] = [],
) -> str:
    """Apply an LLM prompt to documents saved by `parse_html_page`.

    This is where index-free retrieval spends its tokens: the document text goes
    to a second model rather than into this conversation. Keep ranges tight on
    large filings.
    """
    args = {"p": prompt, "r": [r.model_dump() for r in input_character_ranges]}
    if (stop := _too_many_repeats(ctx, "retrieve_information", args)):
        return stop
    emit_tool_start(ctx.deps, "retrieve_information", prompt[:56])

    store = _store(ctx)
    keys = _PLACEHOLDER.findall(prompt)
    if not keys:
        return json.dumps({
            "error": "Your prompt must include at least one data storage key in "
                     "the form {{key_name}}.",
            "data_storage_keys": sorted(store),
            "hint": "Add documents to data storage with parse_html_page.",
        })

    ranges = {r.key: (r.start, r.end) for r in input_character_ranges}
    for key in ranges:
        if key not in set(keys):
            return json.dumps({
                "error": f"The key {key!r} appears in input_character_ranges but "
                         f"not in the prompt.",
                "keys_in_prompt": sorted(set(keys)),
            })
    for key in keys:
        if key not in store:
            return json.dumps({
                "error": f"The key {key!r} was not found in data storage.",
                "data_storage_keys": sorted(store),
                "hint": "Add documents to data storage with parse_html_page.",
            })

    # One substitution pass, so document text is never rescanned: braces inside a
    # filing cannot trigger another key's substitution.
    used: list[tuple[str, int, str]] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        text = store[key]
        start, end = ranges.get(key, (0, len(text)))
        piece = text[start:end]
        used.append((key, start, piece))
        return piece

    filled = _PLACEHOLDER.sub(replace, prompt)

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_openai_base_url,
    )
    try:
        completion = await client.chat.completions.create(
            model=settings.gemini_keyword_model,
            messages=[{"role": "user", "content": filled}],
            temperature=0,
        )
        answer = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        _record_failure(ctx, "retrieve_information")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    # Meter the sub-model. These tokens are billed but sit outside the agent run,
    # so a usage report that ignores them understates this mode substantially.
    usage = getattr(completion, "usage", None)
    if usage is not None:
        ctx.deps.extract_calls += 1
        ctx.deps.extract_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        ctx.deps.extract_output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    cited = [_register(ctx, key, start, piece) for key, start, piece in used]

    result = json.dumps({
        "answer": answer,
        "documents_read": [
            {"key": key, "characters": len(piece), "chunk_id": chunk_id}
            for (key, _, piece), chunk_id in zip(used, cited)
        ],
        "CITE_AS": (
            "Cite a chunk_id above with an excerpt copied verbatim from that "
            "document. Never invent a chunk id."
        ),
    })
    return remember(ctx, "retrieve_information", args, result)
