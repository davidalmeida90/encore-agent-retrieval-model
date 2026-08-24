import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.corpus import router as corpus_router
from app.api.tools import router as tools_router
from app.config import settings

from app import telemetry
from app.agent.progress import add_progress_listener
from app.retrieval.rerank import warm_reranker

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the reranker before the first question arrives.

    Loading torch and the cross-encoder takes several seconds, and lazily it all
    lands on whoever asks first: an observed run spent 75s on a question that
    needs about 9s once warm. Doing it here moves the cost to boot, where nobody
    is waiting. Failure is ignored on purpose, since rerank falls back to fusion
    order and a warm-up problem must not stop the server starting.
    """
    if settings.retrieval_rerank_enabled:
        try:
            await asyncio.to_thread(warm_reranker)
        except Exception:
            pass
    yield


app = FastAPI(title="Encore", lifespan=lifespan)
add_progress_listener(telemetry.progress_listener)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(tools_router)
app.include_router(corpus_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
