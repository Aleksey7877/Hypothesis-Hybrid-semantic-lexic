from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import eval as eval_router
from app.api.routers import ingest as documents_router
from app.api.routers import search
from app.db.postgres import init_db
from app.services.reranking import bge as bge_reranker

log = logging.getLogger("uvicorn.error")


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- DB init ---
    init_db()

    # --- Preload reranker (чтобы не грузить на первом /search) ---
    if _truthy(os.getenv("BGE_RERANK_PRELOAD", "1")) and _truthy(os.getenv("BGE_RERANK_ENABLED", "1")):
        t0 = time.time()
        try:
            bge_reranker.warmup()
            cfg = bge_reranker.get_cfg()
            log.warning(
                "[startup] bge reranker ready model=%s device=%s cache=%s in %.2fs",
                cfg.model_name,
                cfg.device,
                cfg.cache_dir,
                time.time() - t0,
            )
        except Exception:
            log.exception("[startup] bge reranker preload failed (will fallback to no-rerank)")
    else:
        log.warning("[startup] bge reranker preload skipped (BGE_RERANK_PRELOAD=0 or BGE_RERANK_ENABLED=0)")

    yield


app = FastAPI(title="AE Development API", version="0.1.0", lifespan=lifespan)

# CORS (опционально)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
# search: /search
app.include_router(search.router, tags=["search"])

# ingest: /documents/ingest
app.include_router(documents_router.router, prefix="/documents", tags=["documents"])

# eval: /eval (если в eval.py путь уже "/eval" внутри — так и будет)
app.include_router(eval_router.router, tags=["eval"])


@app.get("/healthz")
def healthz():
    # Легкий healthcheck: процесс жив
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    # Readiness: модель реранкера (если включена) готова
    enabled = _truthy(os.getenv("BGE_RERANK_ENABLED", "1"))
    if enabled and not bge_reranker.is_ready():
        return {"status": "starting", "reranker_ready": False, "reranker_enabled": enabled}
    return {"status": "ready", "reranker_ready": bge_reranker.is_ready(), "reranker_enabled": enabled}
