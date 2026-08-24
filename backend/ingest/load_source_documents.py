"""Load normalized Markdown filings from corpus/markdown into source_documents."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import SourceDocument

# Params: edit these, then run `uv run python -m ingest.load_source_documents`
MARKDOWN_DIR = Path(__file__).resolve().parents[2] / "corpus" / "markdown"
SKIP_EXISTING = True

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
}


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def fiscal_year_from_report_date(report_date: str | None) -> int | None:
    if not report_date:
        return None
    return int(report_date[:4])


def load_source_documents() -> dict[str, int]:
    manifest_path = MARKDOWN_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `uv run corpus/convert_to_markdown.py` first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filings = manifest.get("filings", [])
    if not filings:
        raise ValueError(f"No filings listed in {manifest_path}")

    engine = create_engine(settings.sqlalchemy_database_url)
    counts = {"inserted": 0, "skipped": 0, "updated": 0}

    with Session(engine) as session:
        for filing in filings:
            accession_number = filing["accession_number"]
            existing = session.scalar(
                select(SourceDocument).where(
                    SourceDocument.accession_number == accession_number
                )
            )

            if existing and SKIP_EXISTING:
                print(f"Skipping existing {accession_number}")
                counts["skipped"] += 1
                continue

            markdown_path = MARKDOWN_DIR / filing["local_path"]
            if not markdown_path.is_file():
                raise FileNotFoundError(f"Missing Markdown file: {markdown_path}")

            fields = {
                "ticker": filing["ticker"],
                "cik": filing["cik"],
                # Manifest first: download.py records what SEC reports. The map
                # below is only a fallback for manifests written before that.
                "company_name": (
                    filing.get("company_name") or COMPANY_NAMES.get(filing["ticker"])
                ),
                "form": filing["form"],
                "filing_date": parse_date(filing["filing_date"]),
                "report_date": parse_date(filing.get("report_date")),
                "fiscal_year": fiscal_year_from_report_date(filing.get("report_date")),
                "accession_number": accession_number,
                "primary_document": filing["primary_document"],
                "source_url": filing["source_url"],
                "markdown_content": markdown_path.read_text(encoding="utf-8"),
                "ingested_at": datetime.now(UTC),
            }

            if existing:
                print(f"Updating {accession_number}...")
                for key, value in fields.items():
                    setattr(existing, key, value)
                counts["updated"] += 1
            else:
                print(f"Inserting {accession_number}...")
                session.add(SourceDocument(**fields))
                counts["inserted"] += 1

        session.commit()

    return counts


if __name__ == "__main__":
    result = load_source_documents()
    print(
        "Loaded source documents: "
        f"{result['inserted']} inserted, "
        f"{result['updated']} updated, "
        f"{result['skipped']} skipped"
    )
