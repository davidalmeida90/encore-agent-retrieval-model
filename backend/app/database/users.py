"""User record provisioning for Supabase Auth users."""

from __future__ import annotations

from app.auth.dependencies import CurrentUser
from app.database.supabase import get_service_role_client


async def ensure_user(user: CurrentUser) -> None:
    """Upsert the authenticated user into the users table via service role."""
    client = await get_service_role_client()
    await (
        client.table("users")
        .upsert({"id": str(user.id), "email": user.email})
        .execute()
    )
