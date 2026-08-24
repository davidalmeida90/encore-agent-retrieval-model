"""## Chunking: turning a 200-page filing into retrievable pieces

    filing.htm  (SEC EDGAR, ~200 pages of nested tables and styled spans)
         |
         v
    normalize_sec_html         repair words split across styled spans
         |
         v
    docling DocumentConverter  HTML -> structured document tree
         |
         +------------------------------+
         v                              v
    HybridChunker                  extract_sec_tables
    prose, <=512 tokens            our own HTML table parser
         |                              |
         v                              v
    narrative chunks              ONE CHUNK PER TABLE ROW
         |                              |
         +--------------+---------------+
                        v
              _fill_forward_sections     give every chunk its Item number
                        v
                  ChunkRecord[]          text + page + section + metadata
                        v
                  embed + store          Supabase pgvector

## Why a chunk is 512 tokens
Small enough that one chunk is about one idea, so its embedding means something
specific. Embed a whole page and the vector becomes an average of everything on
it, matching every question weakly and none of them well. Small enough also that
six of them fit in a prompt without dominating it.

## Why tables get one chunk PER ROW
A 10-K's numbers live in tables, and a table embedded whole has the same
averaging problem as a page: "Payments for acquisition of property, plant and
equipment | (12,715) | (9,447)" is a precise fact, while the entire cash flow
statement as one vector is a vague gesture at the topic of cash.

So `sec_tables.py` parses tables separately and each row becomes its own chunk,
carrying its table title so the row is readable alone. `_chunk_contains_table`
detects when docling's prose chunk swallowed a table and strips it, to avoid
storing the same numbers twice in two shapes.

## Why the HTML is repaired before docling sees it
EDGAR splits words across adjacent styled spans:

    <span>ITEM 1. B</span><span>USINESS</span>

Any HTML-to-text conversion that separates inline elements emits "B USINESS", and
docling does. Real damage measured in the Microsoft FY2025 filing: B USINESS,
RIS K FACTORS, FLOWS S TATEMENTS, M ICROSOFT, D ELOITTE, and "acqui ition" for
acquisition. 49 occurrences in one document.

Repairing this afterwards is not safely possible: in "RIS K FACTORS" the stray
letter belongs to the word BEFORE it, in "FLOWS S TATEMENTS" to the word AFTER,
and nothing distinguishes the two without a dictionary. Hence the fix at source.
A corrupted word is worse than a missing one, because full-text search for
"acquisition" silently fails to match "acqui ition" and the answer looks merely
unlucky.

## Why sections are filled forward
`_section_from_chunk` looks at ONE chunk: docling's headings if present,
otherwise an "Item 1A."-style marker in the text. Body prose repeats neither, so
about 40% of chunks had no section at all. Records are in document order, so
carrying the last known section forward recovers it. Inherited values are flagged
in metadata rather than being indistinguishable from a real heading.

Result on the current corpus: 31 chunks without a section, out of 27,222.

## What one chunk costs
About 190 tokens of text on average, a 1536-float embedding, and roughly 6KB in
Postgres. Embedding is the cheap part (fractions of a cent per filing); docling
conversion is the slow part, a minute or two per document.

Chunks per filing vary far more than expected, and by industry rather than by
page count. Measured across 23 filings from 10 companies:

    JPM    4,600 per filing   banks disclose enormous loan and risk tables
    GOOGL  1,030
    AAPL     610

JPM alone is 34% of a 27,222-chunk corpus. Worth knowing before assuming a
generic search is sampling companies evenly.

## Known noise
1,063 chunks (3.9%) are table-of-contents rows, e.g.
"Form 10-K Index | Item 3. | Legal Proceedings | page". Every 10-K has an index
table, and each of its rows currently becomes a chunk. Harmless but pointless:
worth filtering on the next ingest by skipping tables whose title matches an
index pattern.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import tiktoken
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from ingest.sec_tables import ExtractedTable, TableRow, extract_sec_tables

# 512 tokens: roughly one idea per chunk, so its embedding means something
# specific rather than averaging a whole page into a vague vector. Also small
# enough that six chunks fit in a prompt without dominating it.
CHUNK_MAX_TOKENS = 512
DOWNLOADS_DIR = Path(__file__).resolve().parents[2] / "corpus" / "downloads"
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "corpus" / "markdown" / "manifest.json"

_ITEM_SECTION_RE = re.compile(r"\bItem\s+[\dA-Z.]+\b", re.IGNORECASE)


class PatchedOpenAITokenizer(OpenAITokenizer):
    """Allow tiktoken special tokens that appear in SEC filing text.

    tiktoken raises on sequences like "<|endoftext|>" unless told not to. Filing
    text is arbitrary and occasionally contains them, and a whole document
    failing to chunk over one stray token sequence is not an acceptable trade.
    """

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text=text,
                allowed_special=set(),
                disallowed_special=(),
            )
        )


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Serialize tables as Markdown for 10-K financial tables."""

    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_index: int
    text: str
    page: str | None
    section: str | None
    token_count: int
    chunk_metadata: dict[str, Any]


def load_manifest_html_paths() -> dict[str, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paths: dict[str, str] = {}
    for filing in manifest.get("filings", []):
        accession = filing["accession_number"]
        html_path = filing.get("html_local_path")
        if not html_path:
            html_path = str(Path(filing["local_path"]).with_suffix(".htm"))
        paths[accession] = html_path
    return paths


def html_path_for_accession(accession_number: str) -> Path:
    paths = load_manifest_html_paths()
    if accession_number not in paths:
        raise KeyError(f"Accession {accession_number} not found in {MANIFEST_PATH}")
    html_path = DOWNLOADS_DIR / paths[accession_number]
    if not html_path.is_file():
        raise FileNotFoundError(f"Missing HTML file: {html_path}")
    return html_path


def build_tokenizer(max_tokens: int = CHUNK_MAX_TOKENS) -> PatchedOpenAITokenizer:
    return PatchedOpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_tokens,
    )


def build_hybrid_chunker(
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> HybridChunker:
    # HybridChunker applies token-aware splits on top of HierarchicalChunker output.
    return HybridChunker(
        tokenizer=build_tokenizer(max_tokens=max_tokens),
        # merge_peers: glue adjacent tiny fragments together. Filings are full of
        # one-line paragraphs, and a 12-token chunk embeds to noise.
        merge_peers=True,
        # repeat_table_header: when a table is split across chunks, repeat its
        # header row in each piece. Without it, "| 12,715 | 9,447 |" arrives with
        # no indication of which years those are.
        repeat_table_header=True,
        # Tables serialise as Markdown rather than flattened text, so the row and
        # column structure survives into the chunk the model eventually reads.
        serializer_provider=MarkdownTableSerializerProvider(),
    )


def build_hierarchical_chunker() -> HierarchicalChunker:
    return HierarchicalChunker(
        serializer_provider=MarkdownTableSerializerProvider(),
    )


# EDGAR renders headings in faux small caps by putting the FIRST letter of a word
# in its own styled tag:
#
#     <span style="font-size:larger">B</span><span>USINESS</span>
#
# Any HTML-to-text conversion that separates inline elements then emits
# "B USINESS", and docling does exactly that. Real damage seen in MSFT FY2025:
#
#     B USINESS   RIS K FACTORS   FLOWS S TATEMENTS   M ICROSOFT   D ELOITTE
#
# Repairing this afterwards is not safely possible: in "RIS K FACTORS" the stray
# letter belongs to the word BEFORE it, in "FLOWS S TATEMENTS" to the word AFTER,
# and no rule tells the two apart without a dictionary. So it is fixed at the
# source, by unwrapping the styling tag before docling ever sees the markup.
#
# Deliberately narrow: only tags whose entire content is ONE letter are unwrapped,
# which is exactly the small-caps artifact. Emphasis tags with real content keep
# their markup, because docling uses them to detect headings.
# EDGAR splits a single word across two ADJACENT styled spans, with the break
# landing mid-word:
#
#     <span style="...">ITEM 1. B</span><span style="...">USINESS</span>
#
# Any HTML-to-text conversion that separates inline elements then emits
# "ITEM 1. B USINESS", and docling does exactly that. Real damage in MSFT FY2025:
#
#     B USINESS   RIS K FACTORS   FLOWS S TATEMENTS   M ICROSOFT   D ELOITTE
#
# Repairing this afterwards is not safely possible: in "RIS K FACTORS" the stray
# letter belongs to the word BEFORE it, in "FLOWS S TATEMENTS" to the word AFTER,
# and nothing distinguishes the two without a dictionary. So it is fixed at the
# source instead.
#
# Only spans with NOTHING between them are merged. Whitespace in the source is a
# real word break and is left alone, so "<span>Total</span> <span>Revenue</span>"
# still yields two words.
_ADJACENT_SPAN_RES = [
    # lone letter ENDS the first span:  "ITEM 1. B</span><span>USINESS"
    (re.compile(r"(\s[A-Za-z])</span><span[^>]*>", re.IGNORECASE), r"\1"),
    (re.compile(r"(\s[A-Za-z])</font><font[^>]*>", re.IGNORECASE), r"\1"),
    # lone letter STARTS the second span: "EXECUTIV</span><span>E OFFICERS"
    (re.compile(r"</span><span[^>]*>([A-Za-z]\s)", re.IGNORECASE), r"\1"),
    (re.compile(r"</font><font[^>]*>([A-Za-z]\s)", re.IGNORECASE), r"\1"),
    # whitespace INSIDE a span edge: "<span> SHEETS</span>". docling trims each
    # span before joining, so the space vanishes and words fuse into
    # "BALANCESHEETS". Hoist it outside the tag where it cannot be trimmed.
    (re.compile(r"<(span|font)([^>]*)>(\s+)", re.IGNORECASE), r"\3<\1\2>"),
    (re.compile(r"(\s+)</(span|font)>", re.IGNORECASE), r"</\2>\1"),
]


def normalize_sec_html(html: str) -> str:
    """Repair words split across adjacent styled spans by EDGAR small-caps markup.

    ONLY a lone letter on one side of the boundary is merged, because that is the
    small-caps artifact. Merging every adjacent span pair (an earlier attempt)
    also glued genuinely separate words together and produced "BALANCESHEETS"
    from "BALANCE SHEETS", trading one corruption for another.
    """
    previous = None
    while previous != html:
        previous = html
        for pattern, repl in _ADJACENT_SPAN_RES:
            html = pattern.sub(repl, html)
    return html


# Docling drops a bare whitespace text node between two adjacent spans, so
# "<span>BALANCE</span> <span>SHEETS</span>" still arrives fused. Hoisting the
# space out of the tag (above) fixes most headings, CASH FLOWS STATEMENTS and
# COMPREHENSIVE INCOME STATEMENTS among them, but not this one.
#
# Rather than keep guessing at docling whitespace rules, the remainder is repaired
# by name. SEC statement headings are a closed set, so this stays a short list
# rather than the dictionary problem that makes general repair unsafe. Anything
# not listed here is left exactly as extracted.
_FUSED_HEADINGS = {
    "BALANCESHEETS": "BALANCE SHEETS",
    "INCOMESTATEMENTS": "INCOME STATEMENTS",
    "CASHFLOWSSTATEMENTS": "CASH FLOWS STATEMENTS",
    "STOCKHOLDERSEQUITYSTATEMENTS": "STOCKHOLDERS EQUITY STATEMENTS",
}


def repair_fused_headings(text: str) -> str:
    for fused, fixed in _FUSED_HEADINGS.items():
        if fused in text:
            text = text.replace(fused, fixed)
    return text


def convert_html_to_document(html_path: Path) -> Any:
    from io import BytesIO

    from docling_core.types.io import DocumentStream

    cleaned = normalize_sec_html(html_path.read_text(encoding="utf-8"))
    stream = DocumentStream(name=html_path.name, stream=BytesIO(cleaned.encode("utf-8")))
    return DocumentConverter().convert(stream).document


def _page_from_chunk_meta(meta: Any) -> str | None:
    origin = getattr(meta, "origin", None)
    if origin is not None:
        page_no = getattr(origin, "page_no", None)
        if page_no is not None:
            return str(page_no)

    for item in getattr(meta, "doc_items", []):
        prov = getattr(item, "prov", None) or []
        for entry in prov:
            page_no = getattr(entry, "page_no", None)
            if page_no is not None:
                return str(page_no)
    return None


def _section_from_chunk(meta: Any, text: str) -> str | None:
    headings = getattr(meta, "headings", None) or []
    if headings:
        return " > ".join(headings)

    match = _ITEM_SECTION_RE.search(text)
    if match:
        return match.group(0)
    return None


def map_chunk_record(
    *,
    chunk_index: int,
    chunk: Any,
    chunker: HybridChunker,
    filing_metadata: dict[str, Any],
) -> ChunkRecord:
    contextualized = chunker.contextualize(chunk=chunk)
    meta = chunk.meta
    tokenizer = chunker.tokenizer

    return ChunkRecord(
        chunk_index=chunk_index,
        text=contextualized,
        page=_page_from_chunk_meta(meta),
        section=_section_from_chunk(meta, contextualized),
        token_count=tokenizer.count_tokens(contextualized),
        chunk_metadata={
            **_base_chunk_metadata(filing_metadata),
            "chunk_kind": "narrative",
            "raw_text": chunk.text,
            "docling_meta": meta.export_json_dict(),
        },
    )


def chunk_document(
    html_path: Path,
    filing_metadata: dict[str, Any],
    *,
    max_chunks: int | None = None,
) -> list[ChunkRecord]:
    html = html_path.read_text(encoding="utf-8")
    doc = convert_html_to_document(html_path)
    chunker = build_hybrid_chunker()
    tables = extract_sec_tables(html)
    used_table_indexes: set[int] = set()
    records: list[ChunkRecord] = []

    for index, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        if max_chunks is not None and index >= max_chunks:
            break
        if _chunk_contains_table(chunk):
            table = _matching_table_for_chunk(
                chunker.contextualize(chunk=chunk),
                tables,
                used_table_indexes,
            )
            narrative_text = _narrative_text_without_tables(
                chunker.contextualize(chunk=chunk)
            )
            if narrative_text:
                records.append(
                    ChunkRecord(
                        chunk_index=len(records),
                        text=narrative_text,
                        page=_page_from_chunk_meta(chunk.meta),
                        section=_section_from_chunk(chunk.meta, narrative_text),
                        token_count=chunker.tokenizer.count_tokens(narrative_text),
                        chunk_metadata={
                            **_base_chunk_metadata(filing_metadata),
                            "chunk_kind": "narrative",
                            "raw_text": narrative_text,
                            "docling_meta": chunk.meta.export_json_dict(),
                        },
                    )
                )
            if table is not None:
                _append_table_row_records(
                    records=records,
                    table=table,
                    chunker=chunker,
                    filing_metadata=filing_metadata,
                )
                used_table_indexes.add(table.table_index)
                continue
            records.append(
                map_chunk_record(
                    chunk_index=len(records),
                    chunk=chunk,
                    chunker=chunker,
                    filing_metadata=filing_metadata,
                )
            )
            continue
        records.append(
            map_chunk_record(
                chunk_index=len(records),
                chunk=chunk,
                chunker=chunker,
                filing_metadata=filing_metadata,
            )
        )

    if max_chunks is None:
        for table in tables:
            if table.table_index in used_table_indexes:
                continue
            _append_table_row_records(
                records=records,
                table=table,
                chunker=chunker,
                filing_metadata=filing_metadata,
            )

    _fill_forward_sections(records)
    return records


def _fill_forward_sections(records: list[ChunkRecord]) -> None:
    """Give sectionless chunks the section of the chunk before them.

    `_section_from_chunk` looks at ONE chunk in isolation: it uses docling's
    headings if there are any, otherwise it looks for an "Item 1A."-style marker
    in the text itself. Body prose repeats neither, so roughly 40% of chunks
    ended up with `section = None` even though their section is obvious from
    position: they sit between a heading and the next one.

    Records are already in document order, so carrying the last known section
    forward recovers it. Inherited values are flagged in `chunk_metadata` rather
    than silently indistinguishable from a heading that was really there, since
    a filing that opens with untitled front matter has no earlier section to
    inherit and legitimately keeps None.
    """
    # repair fused headings before the fill-forward, so an inherited section
    # carries the corrected form rather than propagating the fused one
    for i, record in enumerate(records):
        fixed_text = repair_fused_headings(record.text)
        fixed_section = repair_fused_headings(record.section) if record.section else record.section
        if fixed_text != record.text or fixed_section != record.section:
            records[i] = replace(record, text=fixed_text, section=fixed_section)

    current: str | None = None
    for i, record in enumerate(records):
        if record.section:
            current = record.section
        elif current is not None:
            # ChunkRecord is frozen, so replace the entry rather than mutate it.
            records[i] = replace(
                record,
                section=current,
                chunk_metadata={**record.chunk_metadata, "section_inherited": True},
            )


def _append_table_row_records(
    *,
    records: list[ChunkRecord],
    table: ExtractedTable,
    chunker: HybridChunker,
    filing_metadata: dict[str, Any],
) -> None:
    for row in table.rows:
        text = _table_row_chunk_text(table, row)
        records.append(
            ChunkRecord(
                chunk_index=len(records),
                text=text,
                page=None,
                section=table.title,
                token_count=chunker.tokenizer.count_tokens(text),
                chunk_metadata={
                    **_base_chunk_metadata(filing_metadata),
                    "chunk_kind": "table_row",
                    "table_index": table.table_index,
                    "table_title": table.title,
                    "row_label": row.label,
                    "raw_text": text,
                    "table": table.to_dict(),
                },
            )
        )


def _base_chunk_metadata(filing_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": filing_metadata.get("ticker"),
        "cik": filing_metadata.get("cik"),
        "company_name": filing_metadata.get("company_name"),
        "form": filing_metadata.get("form"),
        "filing_date": filing_metadata.get("filing_date"),
        "report_date": filing_metadata.get("report_date"),
        "fiscal_year": filing_metadata.get("fiscal_year"),
        "accession_number": filing_metadata.get("accession_number"),
        "primary_document": filing_metadata.get("primary_document"),
        "source_url": filing_metadata.get("source_url"),
    }


def _chunk_contains_table(chunk: Any) -> bool:
    for item in getattr(chunk.meta, "doc_items", []) or []:
        label = str(getattr(item, "label", "")).lower()
        if "table" in label:
            return True
    return False


def _matching_table_for_chunk(
    chunk_text: str,
    tables: list[ExtractedTable],
    used_table_indexes: set[int],
) -> ExtractedTable | None:
    for table in tables:
        if table.table_index in used_table_indexes:
            continue
        if _table_matches_chunk(chunk_text, table):
            return table
    return None


def _table_matches_chunk(chunk_text: str, table: ExtractedTable) -> bool:
    if not table.rows:
        return False
    first_row = table.rows[0]
    if first_row.label and first_row.label in chunk_text:
        return True
    return any(cell.text.strip("$") in chunk_text for cell in first_row.cells if cell.text)


def _narrative_text_without_tables(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _table_row_chunk_text(table: ExtractedTable, row: TableRow) -> str:
    title = table.title or f"Table {table.table_index + 1}"
    lines = [title]
    if table.units:
        lines.append(f"Units: {table.units}")

    row_markdown = _markdown_for_row(table, row)
    lines.append(row_markdown)
    if table.footnotes:
        lines.extend(table.footnotes)
    return "\n".join(lines)


def _markdown_for_row(table: ExtractedTable, row: TableRow) -> str:
    header = "| " + " | ".join(column.label for column in table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = "| " + " | ".join([row.label, *[cell.text for cell in row.cells]]) + " |"
    return "\n".join([header, separator, body])


def chunk_document_hierarchical(html_path: Path) -> list[str]:
    """Layout-only chunks from HierarchicalChunker (used in tests / inspection)."""
    doc = convert_html_to_document(html_path)
    chunker = build_hierarchical_chunker()
    return [chunk.text for chunk in chunker.chunk(dl_doc=doc)]


def iter_all_html_paths() -> Iterator[tuple[str, Path]]:
    for accession, relative_path in load_manifest_html_paths().items():
        yield accession, DOWNLOADS_DIR / relative_path
