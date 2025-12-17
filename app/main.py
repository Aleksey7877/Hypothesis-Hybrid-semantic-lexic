from __future__ import annotations

# ВАЖНО: сначала конфиг потоков, потом всё остальное
from app.core.perf import configure_threads
configure_threads(default_cap=8)

import faulthandler
import signal

faulthandler.register(signal.SIGUSR1, all_threads=True)

from fastapi import FastAPI

from app.core.config import settings
from app.db.repository import init_db

from app.api.routers.ingest import router as ingest_router
from app.api.routers.search import router as search_router
from app.api.routers.eval import router as eval_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    @app.on_event("startup")
    def _startup() -> None:
        init_db()

    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(eval_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
