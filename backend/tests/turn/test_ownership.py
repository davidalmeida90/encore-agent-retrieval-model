import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.auth.dependencies import CurrentUser
from app.database.chats import require_thread_access


def _current_user() -> CurrentUser:
    return CurrentUser(id=uuid.uuid4(), email="analyst@example.com")


@pytest.mark.anyio
async def test_require_thread_access_returns_row_for_owner() -> None:
    user = _current_user()
    thread_id = uuid.uuid4()
    mock_response = MagicMock()
    mock_response.data = {
        "id": str(thread_id),
        "user_id": str(user.id),
        "title": "My thread",
    }

    mock_execute = AsyncMock(return_value=mock_response)
    mock_maybe_single = MagicMock()
    mock_maybe_single.execute = mock_execute
    mock_eq = MagicMock()
    mock_eq.maybe_single.return_value = mock_maybe_single
    mock_select = MagicMock()
    mock_select.eq.return_value = mock_eq
    mock_table = MagicMock()
    mock_table.select.return_value = mock_select
    mock_client = MagicMock()
    mock_client.table.return_value = mock_table

    with patch(
        "app.database.chats.get_service_role_client",
        AsyncMock(return_value=mock_client),
    ):
        row = await require_thread_access(thread_id, user)

    assert row.id == thread_id
    assert row.user_id == user.id
    assert row.title == "My thread"


@pytest.mark.anyio
async def test_require_thread_access_raises_404_when_missing() -> None:
    user = _current_user()
    thread_id = uuid.uuid4()
    mock_response = MagicMock()
    mock_response.data = None

    mock_execute = AsyncMock(return_value=mock_response)
    mock_maybe_single = MagicMock()
    mock_maybe_single.execute = mock_execute
    mock_eq = MagicMock()
    mock_eq.maybe_single.return_value = mock_maybe_single
    mock_select = MagicMock()
    mock_select.eq.return_value = mock_eq
    mock_table = MagicMock()
    mock_table.select.return_value = mock_select
    mock_client = MagicMock()
    mock_client.table.return_value = mock_table

    with patch(
        "app.database.chats.get_service_role_client",
        AsyncMock(return_value=mock_client),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await require_thread_access(thread_id, user)

    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_require_thread_access_raises_403_for_other_user() -> None:
    user = _current_user()
    other_user_id = uuid.uuid4()
    thread_id = uuid.uuid4()
    mock_response = MagicMock()
    mock_response.data = {
        "id": str(thread_id),
        "user_id": str(other_user_id),
        "title": "Someone else's thread",
    }

    mock_execute = AsyncMock(return_value=mock_response)
    mock_maybe_single = MagicMock()
    mock_maybe_single.execute = mock_execute
    mock_eq = MagicMock()
    mock_eq.maybe_single.return_value = mock_maybe_single
    mock_select = MagicMock()
    mock_select.eq.return_value = mock_eq
    mock_table = MagicMock()
    mock_table.select.return_value = mock_select
    mock_client = MagicMock()
    mock_client.table.return_value = mock_table

    with patch(
        "app.database.chats.get_service_role_client",
        AsyncMock(return_value=mock_client),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await require_thread_access(thread_id, user)

    assert exc_info.value.status_code == 403
