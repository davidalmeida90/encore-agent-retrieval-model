import uuid
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from app.auth.dependencies import CurrentUser
from app.database.chats import append_grounded_turn, create_thread
from app.schemas.chat import CitationPart, CitationPayload, TextPart, UIMessage


NOW = datetime(2026, 6, 5, 12, 0, 0, tzinfo=UTC)


class FakeInsertBuilder:
    def __init__(self) -> None:
        self.inserted_row: dict[str, Any] | None = None
        self.selected_columns: str | None = None

    def select(self, columns: str) -> "FakeInsertBuilder":
        self.selected_columns = columns
        return self

    async def execute(self) -> SimpleNamespace:
        assert self.inserted_row is not None
        return SimpleNamespace(
            data=[
                {
                    "id": self.inserted_row["id"],
                    "title": self.inserted_row["title"],
                    "created_at": NOW,
                    "updated_at": NOW,
                }
            ]
        )


class FakeTable:
    def __init__(self, builder: FakeInsertBuilder) -> None:
        self.builder = builder

    def insert(self, row: dict[str, Any]) -> FakeInsertBuilder:
        self.builder.inserted_row = row
        return self.builder


class FakeClient:
    def __init__(self) -> None:
        self.builder = FakeInsertBuilder()
        self.table_name: str | None = None

    def table(self, name: str) -> FakeTable:
        self.table_name = name
        return FakeTable(self.builder)


class FakeMutationBuilder:
    def __init__(self, data: list[dict[str, Any]] | None = None) -> None:
        self.data = data or []

    def eq(self, _column: str, _value: str) -> "FakeMutationBuilder":
        return self

    async def execute(self) -> SimpleNamespace:
        return SimpleNamespace(data=self.data)


class FakeSelectBuilder:
    def order(self, _column: str, *, desc: bool = False) -> "FakeSelectBuilder":
        return self

    def limit(self, _count: int) -> "FakeSelectBuilder":
        return self

    def eq(self, _column: str, _value: str) -> "FakeSelectBuilder":
        return self

    async def execute(self) -> SimpleNamespace:
        return SimpleNamespace(data=[])


class FakePersistenceTable:
    def __init__(self, client: "FakePersistenceClient", name: str) -> None:
        self.client = client
        self.name = name

    def select(self, _columns: str) -> FakeSelectBuilder:
        return FakeSelectBuilder()

    def insert(self, rows: list[dict[str, Any]]) -> FakeMutationBuilder:
        self.client.inserted[self.name] = rows
        return FakeMutationBuilder(rows)

    def update(self, row: dict[str, Any]) -> FakeMutationBuilder:
        self.client.updated[self.name] = row
        return FakeMutationBuilder([row])


class FakePersistenceClient:
    def __init__(self) -> None:
        self.inserted: dict[str, list[dict[str, Any]]] = {}
        self.updated: dict[str, dict[str, Any]] = {}

    def table(self, name: str) -> FakePersistenceTable:
        return FakePersistenceTable(self, name)


@pytest.mark.anyio
async def test_create_thread_uses_insert_response_data() -> None:
    user = CurrentUser(id=uuid.uuid4(), email="analyst@example.com")
    client = FakeClient()

    thread = await create_thread(client, user)

    assert client.table_name == "chat_threads"
    assert client.builder.selected_columns == "id,title,created_at,updated_at"
    assert client.builder.inserted_row is not None
    assert client.builder.inserted_row["user_id"] == str(user.id)
    assert client.builder.inserted_row["title"] == "New chat"
    assert thread.id == uuid.UUID(client.builder.inserted_row["id"])
    assert thread.title == "New chat"


@pytest.mark.anyio
async def test_append_grounded_turn_includes_ids_for_citation_rows() -> None:
    chunk_id = uuid.uuid4()
    assistant_message_id = uuid.uuid4()
    client = FakePersistenceClient()

    await append_grounded_turn(
        client,
        thread_id=uuid.uuid4(),
        user_message=UIMessage(
            role="user",
            parts=[TextPart(text="What changed in cloud revenue?")],
        ),
        assistant_message=UIMessage(
            id=str(assistant_message_id),
            role="assistant",
            parts=[
                TextPart(text="Cloud revenue increased [1]."),
                CitationPart(
                    id=str(chunk_id),
                    data=CitationPayload(
                        citation_index=1,
                        chunk_id=chunk_id,
                        excerpt="Cloud revenue increased significantly.",
                        ticker="MSFT",
                        company_name="Microsoft Corporation",
                        form="10-K",
                        filing_date=date(2024, 7, 30),
                        page="15",
                        section="MD&A",
                    ),
                ),
            ],
        ),
        thread_title="Existing chat",
    )

    citation_rows = client.inserted["message_citations"]
    assert len(citation_rows) == 1
    assert uuid.UUID(citation_rows[0]["id"])
    assert citation_rows[0]["message_id"] == str(assistant_message_id)
