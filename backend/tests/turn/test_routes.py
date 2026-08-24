import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import CurrentUser, get_access_token, get_current_user
from app.api.chat import citation_context_response
from app.database.models import DocumentChunk, SourceDocument
from app.main import app
from app.schemas.chat import (
    CitationContextChunk,
    CitationContextResponse,
    CitationPart,
    StreamRequest,
    ThreadResponse,
)

TEST_USER = CurrentUser(id=uuid.uuid4(), email="analyst@example.com")
OTHER_USER = CurrentUser(id=uuid.uuid4(), email="other@example.com")
THREAD_ID = uuid.uuid4()
NOW = datetime(2026, 6, 5, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    app.dependency_overrides[get_access_token] = lambda: "test-token"
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_get_threads_returns_thread_list(client: TestClient) -> None:
    thread = ThreadResponse(id=THREAD_ID, title="Test", created_at=NOW, updated_at=NOW)

    with (
        patch("app.api.chat.ensure_user", AsyncMock()),
        patch("app.api.chat.create_user_client", AsyncMock(return_value=MagicMock())),
        patch("app.api.chat.list_threads", AsyncMock(return_value=[thread])),
    ):
        response = client.get("/chat/threads", headers={"Authorization": "Bearer test"})

    assert response.status_code == 200
    body = response.json()
    assert body["threads"][0]["id"] == str(THREAD_ID)
    assert body["threads"][0]["createdAt"] == NOW.isoformat().replace("+00:00", "Z")


def test_post_thread_creates_thread(client: TestClient) -> None:
    thread = ThreadResponse(id=THREAD_ID, title="New chat", created_at=NOW, updated_at=NOW)

    with (
        patch("app.api.chat.ensure_user", AsyncMock()),
        patch("app.api.chat.create_user_client", AsyncMock(return_value=MagicMock())),
        patch("app.api.chat.create_thread", AsyncMock(return_value=thread)),
    ):
        response = client.post(
            "/chat/threads",
            headers={"Authorization": "Bearer test"},
            json={"title": "New chat"},
        )

    assert response.status_code == 200
    assert response.json()["id"] == str(THREAD_ID)


def test_post_thread_accepts_empty_body(client: TestClient) -> None:
    thread = ThreadResponse(id=THREAD_ID, title="New chat", created_at=NOW, updated_at=NOW)

    with (
        patch("app.api.chat.ensure_user", AsyncMock()),
        patch("app.api.chat.create_user_client", AsyncMock(return_value=MagicMock())),
        patch("app.api.chat.create_thread", AsyncMock(return_value=thread)) as mock_create,
    ):
        response = client.post(
            "/chat/threads",
            headers={"Authorization": "Bearer test"},
            json={},
        )

    assert response.status_code == 200
    mock_create.assert_awaited_once()
    assert mock_create.await_args.kwargs["title"] is None


def test_post_thread_rejects_missing_body(client: TestClient) -> None:
    with (
        patch("app.api.chat.ensure_user", AsyncMock()),
        patch("app.api.chat.create_user_client", AsyncMock(return_value=MagicMock())),
    ):
        response = client.post(
            "/chat/threads",
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 422


def test_delete_thread_removes_owned_thread(client: TestClient) -> None:
    from app.database.chats import ThreadRow

    thread = ThreadRow(id=THREAD_ID, user_id=TEST_USER.id, title="Old chat")

    with (
        patch("app.api.chat.require_thread_access", AsyncMock(return_value=thread)),
        patch("app.api.chat.create_user_client", AsyncMock(return_value=MagicMock())),
        patch("app.api.chat.delete_thread", AsyncMock(), create=True) as mock_delete,
    ):
        response = client.delete(
            f"/chat/threads/{THREAD_ID}",
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 204
    assert response.content == b""
    mock_delete.assert_awaited_once()


def test_delete_thread_returns_403_for_foreign_thread(client: TestClient) -> None:
    from fastapi import HTTPException

    with patch(
        "app.api.chat.require_thread_access",
        AsyncMock(
            side_effect=HTTPException(status_code=403, detail="Forbidden"),
        ),
    ):
        response = client.delete(
            f"/chat/threads/{THREAD_ID}",
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 403


def test_get_messages_returns_403_for_foreign_thread(client: TestClient) -> None:
    from fastapi import HTTPException

    with patch(
        "app.api.chat.require_thread_access",
        AsyncMock(
            side_effect=HTTPException(status_code=403, detail="Forbidden"),
        ),
    ):
        response = client.get(
            f"/chat/threads/{THREAD_ID}/messages",
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 403


def test_get_citation_context_returns_neighbor_chunks(client: TestClient) -> None:
    chunk_id = uuid.uuid4()
    document_id = uuid.uuid4()
    context = CitationContextResponse(
        anchor_chunk_id=chunk_id,
        document_id=document_id,
        ticker="AAPL",
        company_name="Apple Inc.",
        form="10-K",
        filing_date="2024-11-01",
        source_url="https://example.com/aapl-10k",
        chunks=[
            CitationContextChunk(
                chunk_id=uuid.uuid4(),
                chunk_index=36,
                role="previous",
                text="Prior context",
                page="7",
                section="Products",
            ),
            CitationContextChunk(
                chunk_id=chunk_id,
                chunk_index=37,
                role="anchor",
                text="| Segment | 2024 |\n| --- | --- |\n| Services | 96,169 |",
                page="8",
                section="Products",
            ),
            CitationContextChunk(
                chunk_id=uuid.uuid4(),
                chunk_index=38,
                role="next",
                text="Following context",
                page="8",
                section="Products",
            ),
        ],
    )

    with patch("app.api.chat.load_citation_context", return_value=context) as mock_load:
        response = client.get(
            f"/chat/citations/{chunk_id}/context?radius=2",
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["anchorChunkId"] == str(chunk_id)
    assert body["documentId"] == str(document_id)
    assert body["companyName"] == "Apple Inc."
    assert [chunk["role"] for chunk in body["chunks"]] == ["previous", "anchor", "next"]
    assert body["chunks"][1]["chunkIndex"] == 37
    assert body["chunks"][1]["text"].startswith("| Segment |")
    mock_load.assert_called_once_with(chunk_id, 2)


def test_citation_context_response_includes_full_table_for_table_chunk() -> None:
    document_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    document = SourceDocument(
        id=document_id,
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form="10-K",
        filing_date="2025-10-31",
        report_date="2025-09-27",
        fiscal_year=2025,
        accession_number="0000320193-25-000079",
        primary_document="aapl-20250927.htm",
        source_url="https://example.com/aapl",
    )
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        chunk_index=37,
        text="Products and Services Performance\n| Category | 2025 Sales |\n| --- | --- |\n| iPhone | $209,586 |",
        page=None,
        section="Products and Services Performance",
        chunk_metadata={
            "chunk_kind": "table_row",
            "table": {
                "table_index": 2,
                "title": "Products and Services Performance",
                "units": "dollars in millions",
                "markdown": "| Category | 2025 Sales |\n| --- | --- |\n| iPhone | $209,586 |",
                "rows": [],
                "columns": [],
                "footnotes": [],
                "source_html_hash": "abc123",
            },
        },
    )
    chunk.document = document

    context = citation_context_response([chunk], anchor_chunk_id=chunk_id)

    assert context.table is not None
    assert context.table.title == "Products and Services Performance"
    assert context.table.units == "dollars in millions"
    assert context.table.markdown.startswith("| Category |")


def test_get_citation_context_returns_404_when_chunk_is_missing(
    client: TestClient,
) -> None:
    chunk_id = uuid.uuid4()

    with patch("app.api.chat.load_citation_context", return_value=None):
        response = client.get(
            f"/chat/citations/{chunk_id}/context",
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 404


def test_post_stream_returns_event_stream(client: TestClient) -> None:
    from app.database.chats import ThreadRow

    thread = ThreadRow(id=THREAD_ID, user_id=TEST_USER.id, title="New chat")

    async def fake_stream(**kwargs):
        yield 'data: {"type":"text-start","id":"msg-1"}\n\n'
        yield 'data: {"type":"text-end","id":"msg-1"}\n\n'

    with (
        patch("app.api.chat.ensure_user", AsyncMock()),
        patch("app.api.chat.require_thread_access", AsyncMock(return_value=thread)),
        patch("app.api.chat.create_user_client", AsyncMock(return_value=MagicMock())),
        patch("app.api.chat.run_turn", fake_stream),
    ):
        response = client.post(
            "/chat/stream",
            headers={"Authorization": "Bearer test"},
            json={
                "threadId": str(THREAD_ID),
                "messages": [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "text": "Hello"}],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "text-start" in response.text


def test_stream_request_accepts_camel_case_citation_parts() -> None:
    thread_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    request = StreamRequest.model_validate(
        {
            "threadId": str(thread_id),
            "messages": [
                {
                    "id": "assistant-message",
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "text": "Answer with a citation."},
                        {
                            "type": "data-citation",
                            "id": str(chunk_id),
                            "data": {
                                "citationIndex": 1,
                                "chunkId": str(chunk_id),
                                "excerpt": "Relevant excerpt",
                                "ticker": "AAPL",
                                "companyName": "Apple Inc.",
                                "form": "10-K",
                                "filingDate": "2024-10-31",
                                "page": "42",
                                "section": "Risk Factors",
                            },
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Follow up"}],
                },
            ],
        }
    )

    citation_part = request.messages[0].parts[1]
    assert isinstance(citation_part, CitationPart)
    assert citation_part.data.citation_index == 1
    assert citation_part.data.chunk_id == chunk_id
    assert citation_part.data.company_name == "Apple Inc."
