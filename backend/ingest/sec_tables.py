"""SEC HTML table extraction for financial filing tables."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

_AMOUNT_RE = re.compile(r"^\(?\$?[\d,]+(?:\.\d+)?\)?$")
_FOOTNOTE_RE = re.compile(r"^\(\d+\)")
_UNIT_RE = re.compile(r"\(([^)]*(?:million|thousand|billion)[^)]*)\)", re.IGNORECASE)
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param"}


@dataclass(frozen=True, slots=True)
class InlineFact:
    name: str | None
    context_ref: str | None
    unit_ref: str | None
    decimals: str | None
    scale: str | None
    fact_id: str | None
    value: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "context_ref": self.context_ref,
            "unit_ref": self.unit_ref,
            "decimals": self.decimals,
            "scale": self.scale,
            "fact_id": self.fact_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class TableColumn:
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label}


@dataclass(frozen=True, slots=True)
class TableCell:
    text: str
    facts: tuple[InlineFact, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "facts": [fact.to_dict() for fact in self.facts],
        }


@dataclass(frozen=True, slots=True)
class TableRow:
    label: str
    cells: tuple[TableCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class ExtractedTable:
    table_index: int
    title: str | None
    units: str | None
    columns: tuple[TableColumn, ...]
    rows: tuple[TableRow, ...]
    footnotes: list[str]
    markdown: str
    source_html_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_index": self.table_index,
            "title": self.title,
            "units": self.units,
            "columns": [column.to_dict() for column in self.columns],
            "rows": [row.to_dict() for row in self.rows],
            "footnotes": self.footnotes,
            "markdown": self.markdown,
            "source_html_hash": self.source_html_hash,
        }


@dataclass(slots=True)
class _Node:
    tag: str
    attrs: dict[str, str]
    children: list[_Node | str] = field(default_factory=list)
    parent: _Node | None = None


@dataclass(frozen=True, slots=True)
class _RawCell:
    text: str
    colspan: int
    facts: tuple[InlineFact, ...]


class _TreeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node(tag="document", attrs={})
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self._stack[-1].children.append("\n")
            return

        node = _Node(
            tag=tag.lower(),
            attrs={key.lower(): value or "" for key, value in attrs},
            parent=self._stack[-1],
        )
        self._stack[-1].children.append(node)
        if tag.lower() not in _VOID_TAGS:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == normalized:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].children.append(data)


def extract_sec_tables(html: str) -> list[ExtractedTable]:
    parser = _TreeParser()
    parser.feed(html)
    html_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()

    tables: list[ExtractedTable] = []
    for table_node in _iter_nodes(parser.root, "table"):
        raw_rows = _raw_rows(table_node)
        if not _is_meaningful(raw_rows):
            continue

        context = _table_context(table_node)
        normalized = _normalize_rows(raw_rows)
        if normalized is None:
            continue

        columns, rows = normalized
        markdown = _to_markdown(columns, rows, context["footnotes"])
        tables.append(
            ExtractedTable(
                table_index=len(tables),
                title=context["title"],
                units=context["units"],
                columns=tuple(columns),
                rows=tuple(rows),
                footnotes=context["footnotes"],
                markdown=markdown,
                source_html_hash=html_hash,
            )
        )

    return tables


def tables_to_json(tables: list[ExtractedTable]) -> list[dict[str, Any]]:
    return [table.to_dict() for table in tables]


def tables_to_markdown(tables: list[ExtractedTable]) -> str:
    blocks: list[str] = []
    for table in tables:
        lines: list[str] = []
        if table.title:
            lines.append(f"## {table.title}")
        else:
            lines.append(f"## Table {table.table_index + 1}")
        if table.units:
            lines.append(f"_Units: {table.units}_")
        lines.append(table.markdown)
        blocks.append("\n\n".join(lines))
    return "\n\n".join(blocks)


def _iter_nodes(node: _Node, tag: str) -> list[_Node]:
    matches: list[_Node] = []
    for child in node.children:
        if isinstance(child, _Node):
            if child.tag == tag:
                matches.append(child)
            matches.extend(_iter_nodes(child, tag))
    return matches


# Tags that interrupt the flow of a word. Everything else is inline styling, and
# EDGAR uses inline styling constantly: a heading's leading capital is wrapped in
# its own <span> for small-caps, and words get split across <b>/<i> runs.
#
# Joining every child with a space (the previous behaviour) corrupted those words
# on the way into the corpus, permanently:
#
#     CASH FLOWS <span>S</span>TATEMENTS  ->  "CASH FLOWS S TATEMENTS"
#     acqui<span>s</span>ition            ->  "acqui s ition"
#
# No amount of retrieval tuning recovers text that was broken at ingest, and a
# keyword search for "acquisition" cannot match "acqui s ition".
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "caption", "dd", "div",
    "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2",
    "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p",
    "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}


def _node_text(node: _Node) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        else:
            inner = _node_text(child)
            # A block boundary is a real separator; inline styling is not, so it
            # is concatenated directly and the word survives intact.
            if child.tag in _BLOCK_TAGS:
                parts.append(f" {inner} ")
            else:
                parts.append(inner)
    return _clean_text("".join(parts))


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _raw_rows(table_node: _Node) -> list[list[_RawCell]]:
    rows: list[list[_RawCell]] = []
    for row_node in _iter_direct_or_nested(table_node, "tr"):
        cells: list[_RawCell] = []
        for cell_node in _cell_nodes(row_node):
            cells.append(
                _RawCell(
                    text=_clean_cell_text(_node_text(cell_node)),
                    colspan=_positive_int(cell_node.attrs.get("colspan"), default=1),
                    facts=tuple(_facts(cell_node)),
                )
            )
        if cells:
            rows.append(cells)
    return rows


def _iter_direct_or_nested(node: _Node, tag: str) -> list[_Node]:
    matches: list[_Node] = []
    for child in node.children:
        if not isinstance(child, _Node):
            continue
        if child.tag == tag:
            matches.append(child)
        elif child.tag in {"thead", "tbody", "tfoot"}:
            matches.extend(_iter_direct_or_nested(child, tag))
    return matches


def _cell_nodes(row_node: _Node) -> list[_Node]:
    return [
        child
        for child in row_node.children
        if isinstance(child, _Node) and child.tag in {"td", "th"}
    ]


def _positive_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return max(int(value), 1)
    except ValueError:
        return default


def _facts(node: _Node) -> list[InlineFact]:
    facts: list[InlineFact] = []
    for child in node.children:
        if not isinstance(child, _Node):
            continue
        if child.tag in {"ix:nonfraction", "ix:nonnumeric", "nonfraction", "nonnumeric"}:
            facts.append(
                InlineFact(
                    name=child.attrs.get("name"),
                    context_ref=child.attrs.get("contextref"),
                    unit_ref=child.attrs.get("unitref"),
                    decimals=child.attrs.get("decimals"),
                    scale=child.attrs.get("scale"),
                    fact_id=child.attrs.get("id"),
                    value=_node_text(child),
                )
            )
        facts.extend(_facts(child))
    return facts


def _clean_cell_text(text: str) -> str:
    if text in {"—", "–"}:
        return "-"
    text = text.replace("$ ", "$")
    text = re.sub(r"\(\s+([\d,]+)\s+\)", r"(\1)", text)
    return text


def _is_meaningful(rows: list[list[_RawCell]]) -> bool:
    texts = [cell.text for row in rows for cell in row if cell.text]
    if len(texts) < 2:
        return False
    return any(_looks_numeric(text) for text in texts)


def _looks_numeric(text: str) -> bool:
    return bool(_AMOUNT_RE.match(text) or re.match(r"^\(?\d+\)?%?$", text))


def _table_context(table_node: _Node) -> dict[str, Any]:
    anchor = _context_anchor(table_node)
    previous = _nearby_text(anchor, direction=-1, limit=4)
    following = _nearby_text(anchor, direction=1, limit=4)
    title = _title_from_previous(previous)
    units = _units_from_text(previous)
    footnotes = [text for text in following if _FOOTNOTE_RE.match(text)]
    return {"title": title, "units": units, "footnotes": footnotes}


def _context_anchor(table_node: _Node) -> _Node:
    parent = table_node.parent
    if parent is None:
        return table_node
    element_children = [child for child in parent.children if isinstance(child, _Node)]
    if element_children == [table_node]:
        return parent
    return table_node


def _nearby_text(node: _Node, *, direction: int, limit: int) -> list[str]:
    parent = node.parent
    if parent is None:
        return []

    siblings = parent.children
    index = next(index for index, sibling in enumerate(siblings) if sibling is node)
    step = -1 if direction < 0 else 1
    texts: list[str] = []
    cursor = index + step
    while 0 <= cursor < len(siblings) and len(texts) < limit:
        sibling = siblings[cursor]
        if isinstance(sibling, _Node) and sibling.tag == "table":
            break
        if isinstance(sibling, _Node):
            text = _node_text(sibling)
            if text:
                texts.append(text)
        cursor += step
    if direction < 0:
        texts.reverse()
    return texts


def _title_from_previous(texts: list[str]) -> str | None:
    for text in reversed(texts):
        if _UNIT_RE.search(text):
            continue
        if text.lower().startswith("the following table"):
            continue
        return text
    return None


def _units_from_text(texts: list[str]) -> str | None:
    for text in reversed(texts):
        match = _UNIT_RE.search(text)
        if match:
            return match.group(1)
    return None


def _normalize_rows(
    raw_rows: list[list[_RawCell]],
) -> tuple[list[TableColumn], list[TableRow]] | None:
    header_row = _first_text_row(raw_rows)
    if header_row is None:
        return None

    header_tokens = _nonempty_texts(header_row)
    data_rows = raw_rows[raw_rows.index(header_row) + 1 :]
    if "Change" in header_tokens:
        return _normalize_sales_change_table(header_tokens, data_rows)
    return _normalize_simple_table(header_tokens, data_rows)


def _first_text_row(rows: list[list[_RawCell]]) -> list[_RawCell] | None:
    for row in rows:
        if len(_nonempty_texts(row)) >= 2:
            return row
    return None


def _nonempty_texts(row: list[_RawCell]) -> list[str]:
    return [cell.text for cell in row if cell.text]


def _normalize_sales_change_table(
    header_tokens: list[str],
    raw_rows: list[list[_RawCell]],
) -> tuple[list[TableColumn], list[TableRow]] | None:
    data_headers: list[str] = []
    last_year: str | None = None
    for token in header_tokens:
        if re.fullmatch(r"\d{4}", token):
            last_year = token
            data_headers.append(f"{token} Sales")
        elif token == "Change" and last_year:
            data_headers.append(f"{last_year} Change")

    columns = [TableColumn("Category"), *[TableColumn(label) for label in data_headers]]
    rows: list[TableRow] = []
    for raw_row in raw_rows:
        row = _sales_change_row(raw_row, expected_values=len(data_headers))
        if row is not None:
            rows.append(row)

    if not rows:
        return None
    return columns, rows


def _sales_change_row(row: list[_RawCell], *, expected_values: int) -> TableRow | None:
    tokens = [cell for cell in row if cell.text]
    if len(tokens) < 2:
        return None

    label = tokens[0].text
    values: list[TableCell] = []
    index = 1
    while index < len(tokens) and len(values) < expected_values:
        if _next_value_is_change(values):
            text, consumed = _consume_change(tokens[index:])
        else:
            text, consumed = _consume_amount(tokens[index:])
        if not text or consumed == 0:
            index += 1
            continue
        facts = tuple(fact for cell in tokens[index : index + consumed] for fact in cell.facts)
        values.append(TableCell(text=text, facts=facts))
        index += consumed

    if len(values) != expected_values:
        return None
    return TableRow(label=label, cells=tuple(values))


def _next_value_is_change(values: list[TableCell]) -> bool:
    return len(values) % 2 == 1


def _consume_amount(cells: list[_RawCell]) -> tuple[str | None, int]:
    if not cells:
        return None, 0
    if cells[0].text == "$" and len(cells) > 1:
        return f"${cells[1].text}", 2
    if cells[0].text.startswith("$"):
        return cells[0].text, 1
    if _AMOUNT_RE.match(cells[0].text):
        return cells[0].text, 1
    return None, 0


def _consume_change(cells: list[_RawCell]) -> tuple[str | None, int]:
    if cells and cells[0].text.endswith("%"):
        return cells[0].text, 1

    collected: list[str] = []
    for cell in cells:
        if cell.text == "%":
            break
        if not collected or cell.text != collected[-1]:
            collected.append(cell.text)
    if not collected:
        return None, 0

    consumed = 0
    for cell in cells:
        consumed += 1
        if cell.text == "%":
            break

    if collected == ["-"]:
        return "-", consumed
    return f"{collected[0]}%", consumed


def _normalize_simple_table(
    header_tokens: list[str],
    raw_rows: list[list[_RawCell]],
) -> tuple[list[TableColumn], list[TableRow]] | None:
    columns = _simple_columns(header_tokens)
    expected_values = len(columns) - 1
    rows: list[TableRow] = []
    for raw_row in raw_rows:
        tokens = [cell for cell in raw_row if cell.text]
        if len(tokens) < 2:
            continue
        label = tokens[0].text
        values: list[TableCell] = []
        index = 1
        while index < len(tokens) and len(values) < expected_values:
            text, consumed = _consume_amount(tokens[index:])
            if not text:
                text = _normalize_value_cell(tokens[index].text)
                consumed = 1
            facts = tuple(fact for cell in tokens[index : index + consumed] for fact in cell.facts)
            values.append(TableCell(text=text, facts=facts))
            index += consumed
        if len(values) == expected_values:
            rows.append(TableRow(label=label, cells=tuple(values)))

    if not rows:
        return None
    return columns, rows


def _simple_columns(header_tokens: list[str]) -> list[TableColumn]:
    if header_tokens and not _looks_like_value_header(header_tokens[0]):
        return [TableColumn(label) for label in header_tokens]
    return [TableColumn("Metric"), *[TableColumn(label) for label in header_tokens]]


def _looks_like_value_header(text: str) -> bool:
    if re.fullmatch(r"\d{4}", text):
        return True
    if re.search(r"\b\d{4}\b", text):
        return True
    return text.lower() in {"amount", "value"}


def _normalize_value_cell(text: str) -> str:
    return text.replace("$ ", "$")


def _to_markdown(
    columns: list[TableColumn],
    rows: list[TableRow],
    footnotes: list[str],
) -> str:
    header = "| " + " | ".join(column.label for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join([row.label, *[cell.text for cell in row.cells]])
        + " |"
        for row in rows
    ]
    lines = [header, separator, *body]
    if footnotes:
        lines.extend(["", *footnotes])
    return "\n".join(lines)
