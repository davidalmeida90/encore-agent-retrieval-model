# Data

Local data artifacts for development live here.

- `downloads/` holds raw source files fetched from SEC EDGAR, grouped by year.
- `markdown/` holds Docling-converted Markdown exports with the same year layout and manifest.
- Downloaded and converted payloads are gitignored because the corpus can get large.
- Fetch a sample corpus with `uv run corpus/download.py`
- Convert HTML filings to Markdown with `uv run corpus/convert_to_markdown.py`
