# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "docling==2.96.0",
# ]
# ///
from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from docling.document_converter import DocumentConverter

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ingest.sec_tables import extract_sec_tables, tables_to_json, tables_to_markdown


# Params: edit these, then run `uv run corpus/convert_to_markdown.py`
INPUT_DIR = Path(__file__).resolve().parent / "downloads"
OUTPUT_DIR = Path(__file__).resolve().parent / "markdown"
CLEAR_OUTPUT_DIR = False
SKIP_EXISTING = True


def convert_downloads_to_markdown() -> dict:
    manifest_path = INPUT_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `uv run corpus/download.py` first."
        )

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filings = source_manifest.get("filings", [])
    if not filings:
        raise ValueError(f"No filings listed in {manifest_path}")

    if CLEAR_OUTPUT_DIR and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    manifest = {
        "source": source_manifest.get("source", "SEC EDGAR"),
        "converted_at_utc": datetime.now(UTC).isoformat(),
        "form": source_manifest.get("form", "10-K"),
        "converted_count": 0,
        "filings": [],
    }

    for filing in filings:
        html_relative = filing["local_path"]
        html_path = INPUT_DIR / html_relative
        if not html_path.is_file():
            raise FileNotFoundError(f"Missing HTML file: {html_path}")

        md_relative = str(Path(html_relative).with_suffix(".md"))
        md_path = OUTPUT_DIR / md_relative
        md_path.parent.mkdir(parents=True, exist_ok=True)
        tables_json_path = md_path.with_suffix(".tables.json")
        html = html_path.read_text(encoding="utf-8")
        tables = extract_sec_tables(html)

        if SKIP_EXISTING and md_path.exists():
            print(f"Skipping existing {md_relative}")
            if tables and "# Normalized Tables" not in md_path.read_text(encoding="utf-8"):
                with md_path.open("a", encoding="utf-8") as output:
                    output.write(f"\n\n# Normalized Tables\n\n{tables_to_markdown(tables)}")
            if tables and not tables_json_path.exists():
                tables_json_path.write_text(
                    json.dumps(tables_to_json(tables), indent=2) + "\n",
                    encoding="utf-8",
                )
        else:
            print(f"Converting {html_relative}...")
            result = converter.convert(html_path)
            markdown = result.document.export_to_markdown()
            if tables:
                markdown = f"{markdown}\n\n# Normalized Tables\n\n{tables_to_markdown(tables)}"
            md_path.write_text(
                markdown,
                encoding="utf-8",
            )
            if tables:
                tables_json_path.write_text(
                    json.dumps(tables_to_json(tables), indent=2) + "\n",
                    encoding="utf-8",
                )

        manifest_filing = {
            **filing,
            "html_local_path": html_relative,
            "local_path": md_relative,
        }
        if tables_json_path.exists():
            manifest_filing["tables_local_path"] = str(
                Path(md_relative).with_suffix(".tables.json")
            )
        manifest["filings"].append(manifest_filing)
        manifest["converted_count"] += 1

    output_manifest_path = OUTPUT_DIR / "manifest.json"
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    result = convert_downloads_to_markdown()
    print(
        f"Converted {result['converted_count']} filing(s) from {INPUT_DIR} to {OUTPUT_DIR}"
    )
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")
