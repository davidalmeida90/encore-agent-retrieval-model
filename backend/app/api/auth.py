from fastapi import APIRouter, Depends

from app.auth.dependencies import CurrentUser, get_current_user

router = APIRouter(tags=["auth"])


@router.get("/me")
async def me(user: CurrentUser = Depends(get_current_user)) -> dict[str, str]:
    return {"id": str(user.id), "email": user.email}
