from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.retrieval.keywords import FtsKeywordExtraction, extract_fts_keywords
from app.retrieval.types import SearchFilters

LONG_NVDA_QUERY = (
    "How did NVIDIA describe demand drivers and customer concentration "
    "for its Data Center business?"
)

LONG_NEUTRAL_QUERY = (
    "What did management say about operating margins and cost structure "
    "across recent fiscal years?"
)


def _mock_parse_response(terms: list[str]) -> MagicMock:
    message = MagicMock()
    message.parsed = FtsKeywordExtraction(terms=terms)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("app.retrieval.keywords._client")
def test_extract_fts_keywords_returns_llm_terms(mock_client: MagicMock) -> None:
    mock_client.return_value.chat.completions.parse.return_value = _mock_parse_response(
        ["data center", "demand", "customer concentration"]
    )

    result = extract_fts_keywords(LONG_NVDA_QUERY, filters=SearchFilters(ticker="NVDA"))

    words = result.split()
    assert len(words) <= 5
    assert "demand" in result.casefold()
    assert "center" in result.casefold()
    assert "nvidia" not in result.casefold()
    call_kwargs = mock_client.return_value.chat.completions.parse.call_args.kwargs
    user_message = call_kwargs["messages"][1]["content"]
    assert "Ticker filter: NVDA" in user_message
    assert LONG_NVDA_QUERY in user_message


@patch("app.retrieval.keywords._client")
def test_extract_fts_keywords_fast_path_skips_llm(mock_client: MagicMock) -> None:
    short_query = "Azure AI cloud capacity"

    result = extract_fts_keywords(short_query)

    assert result == short_query
    mock_client.return_value.chat.completions.parse.assert_not_called()


@patch("app.retrieval.keywords._client")
def test_extract_fts_keywords_clamps_to_max_terms(mock_client: MagicMock) -> None:
    mock_client.return_value.chat.completions.parse.return_value = _mock_parse_response(
        ["one", "two", "three", "four", "five", "six"]
    )

    result = extract_fts_keywords(LONG_NEUTRAL_QUERY)

    assert result == "one two three four five"


@patch("app.retrieval.keywords._client")
def test_extract_fts_keywords_dedupes_terms(mock_client: MagicMock) -> None:
    mock_client.return_value.chat.completions.parse.return_value = _mock_parse_response(
        ["Azure", "azure", "AI", "cloud"]
    )

    result = extract_fts_keywords(LONG_NEUTRAL_QUERY)

    assert result == "Azure AI cloud"


@patch("app.retrieval.keywords._client")
def test_extract_fts_keywords_falls_back_when_llm_raises(mock_client: MagicMock) -> None:
    mock_client.return_value.chat.completions.parse.side_effect = RuntimeError("api down")

    result = extract_fts_keywords(LONG_NVDA_QUERY)

    assert "NVIDIA" in result
    assert "demand" in result.casefold()
    assert len(result.split()) <= 5
    assert "how" not in result.casefold().split()


@patch("app.retrieval.keywords._client")
def test_extract_fts_keywords_falls_back_when_too_few_terms(mock_client: MagicMock) -> None:
    mock_client.return_value.chat.completions.parse.return_value = _mock_parse_response(
        ["Azure"]
    )

    result = extract_fts_keywords(LONG_NVDA_QUERY)

    assert len(result.split()) >= 3


@patch("app.retrieval.keywords._client")
def test_extract_fts_keywords_falls_back_when_parse_is_none(mock_client: MagicMock) -> None:
    message = MagicMock()
    message.parsed = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    mock_client.return_value.chat.completions.parse.return_value = response

    result = extract_fts_keywords(LONG_NVDA_QUERY)

    assert result
    assert "how" not in result.casefold().split()
