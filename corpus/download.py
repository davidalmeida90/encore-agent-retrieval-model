# /// script
# requires-python = ">=3.12"
# ///
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib import request


# Params: edit these, then run `uv run corpus/download.py`
USER_AGENT = os.getenv("SEC_EDGAR_UA", "Encore research@example.com")
# Ten large caps with a deliberate sector spread, so comparison questions have
# something to compare. Tech alone would make every "compare X and Y" question a
# variation on the same answer.
TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # Nvidia
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet
    "META",   # Meta
    "TSLA",   # Tesla
    "NFLX",   # Netflix
    "JPM",    # JPMorgan, so not every comparison is a tech comparison
    "WMT",    # Walmart, likewise
]
# 3 per company: Microsoft filed FY2026 on 2026-07-29 and 2 was leaving it out,
# so "most recent fiscal year" meant Apple FY2025 vs Microsoft FY2026 while the
# corpus only held Microsoft FY2025. XBRL and the filings disagreed, and the model
# looped trying to reconcile them.
FILINGS_PER_COMPANY = 3
OUTPUT_DIR = Path(__file__).resolve().parent / "downloads"
# leave existing downloads in place; same-named files are overwritten
CLEAR_OUTPUT_DIR = False

# CIKs are resolved from the SEC's own ticker file rather than hardcoded, so
# changing TICKERS above is all that is needed to change the corpus. The previous
# hardcoded map covered five tickers and silently had no answer for a sixth.
_SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

# Ticker -> CIK is not always the entity that files. Exxon reorganised into a
# holding company: "ExxonMobil Holdings Corp" (2115436) now carries the XOM
# ticker and has filed no 10-K, while "EXXON MOBIL CORP" (34088) has every
# filing and no ticker. Resolution therefore succeeded, found nothing, and the
# download quietly produced 9 companies out of 10.
#
# Override when the ticker points at the wrong entity.
CIK_OVERRIDES = {
    "XOM": "0000034088",
}
_cik_cache: dict[str, str] = {}


def cik_for(ticker: str) -> str:
    """Return the zero-padded 10-digit CIK for a ticker, from SEC's mapping."""
    if not _cik_cache:
        data = get_json(_SEC_TICKER_URL)
        for row in data.values():
            _cik_cache[str(row["ticker"]).upper()] = f"{int(row['cik_str']):010d}"
    key = ticker.upper()
    if key in CIK_OVERRIDES:
        return CIK_OVERRIDES[key]
    if key not in _cik_cache:
        raise KeyError(f"No CIK found for {ticker!r} in SEC company_tickers.json")
    return _cik_cache[key]


def get_json(url: str) -> dict:
    req = request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(url: str) -> bytes:
    req = request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "User-Agent": USER_AGENT,
        },
    )
    with request.urlopen(req, timeout=60) as response:
        return response.read()


def download_filings() -> dict:
    if CLEAR_OUTPUT_DIR and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # range() is END-EXCLUSIVE, so the original bound of datetime.now().year
    # silently dropped the CURRENT year and this script could never fetch a
    # filing made this year. Microsoft filed FY2026 on 2026-07-29 and it was
    # skipped every run, which left XBRL a year ahead of the corpus.
    this_year = datetime.now(UTC).year
    target_years = {
        str(year)
        for year in range(this_year - FILINGS_PER_COMPANY + 1, this_year + 1)
    }
    manifest = {
        "source": "SEC EDGAR",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "form": "10-K",
        "downloaded_count": 0,
        "filings": [],
    }

    for ticker in TICKERS:
        print(f"Downloading {ticker} filings...")
        cik = cik_for(ticker)
        submission = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
        # SEC already knows the legal name, so record it rather than keeping a
        # hand-written ticker -> name map somewhere else. The map version covered
        # 5 of 10 tickers, because the ticker list grew and it did not.
        company_name = submission.get("name") or None
        submissions = [submission]
        submissions.extend(
            get_json(f"https://data.sec.gov/submissions/{item['name']}")
            for item in submission.get("filings", {}).get("files", [])
        )

        filings = []
        for sec_submission in submissions:
            filings.extend(extract_10k_filings(sec_submission, target_years))
            if len(filings) >= FILINGS_PER_COMPANY:
                break

        # A ticker that resolves but files nothing must not pass quietly. XOM did
        # exactly that (its ticker moved to a holding company with no 10-K) and
        # the corpus silently ended up one company short.
        if not filings:
            print(
                f"  WARNING: {ticker} (CIK {cik}) returned no 10-K in "
                f"{sorted(target_years)}. Check CIK_OVERRIDES.",
                flush=True,
            )

        for filing in filings[:FILINGS_PER_COMPANY]:
            accession_path = filing["accession_number"].replace("-", "")
            source_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_path}/{filing['primary_document']}"
            )
            year_dir = OUTPUT_DIR / filing["year"]
            year_dir.mkdir(parents=True, exist_ok=True)
            local_path = year_dir / (
                f"{ticker.lower()}_{filing['form'].lower()}_"
                f"{filing['filing_date']}_{filing['accession_number']}"
                f"{Path(filing['primary_document']).suffix or '.html'}"
            )
            local_path.write_bytes(get_bytes(source_url))

            manifest["filings"].append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": company_name,
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "report_date": filing["report_date"],
                    "accession_number": filing["accession_number"],
                    "primary_document": filing["primary_document"],
                    "source_url": source_url,
                    "local_path": str(local_path.relative_to(OUTPUT_DIR)),
                }
            )
            manifest["downloaded_count"] += 1

            time.sleep(0.2)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def extract_10k_filings(
    submission: dict, target_years: set[str]
) -> list[dict[str, str]]:
    recent = submission["filings"]["recent"] if "filings" in submission else submission
    filings = []

    for form, accession, document, filing_date, report_date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"],
        recent["reportDate"],
        strict=True,
    ):
        year = (report_date or filing_date)[:4]
        if form == "10-K" and year in target_years:
            filings.append(
                {
                    "year": year,
                    "form": form,
                    "accession_number": accession,
                    "primary_document": document,
                    "filing_date": filing_date,
                    "report_date": report_date,
                }
            )

    return filings


if __name__ == "__main__":
    result = download_filings()
    print(f"Downloaded {result['downloaded_count']} filing(s) to {OUTPUT_DIR}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")
