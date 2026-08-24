"""Supabase client construction for server-side database and auth access."""

from supabase import AsyncClient, acreate_client
from supabase.lib.client_options import AsyncClientOptions

from app.config import settings

_service_role_client: AsyncClient | None = None


def _server_client_options(
    *,
    access_token: str | None = None,
) -> AsyncClientOptions:
    headers: dict[str, str] = {}
    if access_token is not None:
        headers["Authorization"] = f"Bearer {access_token}"

    return AsyncClientOptions(
        headers=headers,
        auto_refresh_token=False,
        persist_session=False,
    )


def _normalize_access_token(access_token: str) -> str:
    prefix = "Bearer "
    if access_token.startswith(prefix):
        return access_token.removeprefix(prefix)
    return access_token


async def get_service_role_client() -> AsyncClient:
    """Return a shared client that bypasses RLS — backend-only privileged access."""
    global _service_role_client

    if _service_role_client is None:
        _service_role_client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
            options=_server_client_options(),
        )

    return _service_role_client


async def create_user_client(access_token: str) -> AsyncClient:
    """Return a request-scoped client that enforces RLS for the authenticated user."""
    return await acreate_client(
        settings.supabase_url,
        settings.supabase_anon_key,
        options=_server_client_options(
            access_token=_normalize_access_token(access_token),
        ),
    )
