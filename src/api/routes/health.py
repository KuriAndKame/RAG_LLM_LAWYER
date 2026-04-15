from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """
    Простой эндпоинт для проверки, что сервер работает.
    """
    return {"status": "ok"}
