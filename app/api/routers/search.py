from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.models import ParentChunk
from app.db.postgres import SessionLocal
from app.db.repository import Repository

log = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/search", tags=["search"])
repo = Repository()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    parent_id: str
    score: float = 0.0
    title: str = ""
    anchor: str = ""
    section_path: str = ""
    part_no: int = 1
    doc_id: str = ""
    content: str = ""


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]


def _fallback_postgres_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Fallback: простой ILIKE по ParentChunk.content.
    """
    q = f"%{query}%"
    with SessionLocal() as s:
        stmt = (
            select(
                ParentChunk.parent_id,
                ParentChunk.doc_id,
                ParentChunk.title,
                ParentChunk.anchor,
                ParentChunk.section_path,
                ParentChunk.part_no,
                ParentChunk.content,
            )
            .where(ParentChunk.content.ilike(q))
            .limit(top_k)
        )
        rows = s.execute(stmt).all()

    return [
        {
            "parent_id": r.parent_id,
            "score": 0.0,
            "title": r.title,
            "anchor": r.anchor,
            "section_path": r.section_path,
            "part_no": r.part_no,
            "doc_id": r.doc_id,
            "content": r.content,
        }
        for r in rows
    ]


@lru_cache(maxsize=1)
def _qdrant_store():
    # кешируем, чтобы модель не грузилась на каждый запрос
    from app.services.embeddings.e5 import E5Embedder
    from app.services.vectorstore.qdrant import QdrantStore

    return QdrantStore(E5Embedder())


def _qdrant_parent_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Ищем в Qdrant по child_chunks, затем агрегируем в parent_hits:
    - дедуп по parent_id
    - берём максимальный score
    """
    store = _qdrant_store()

    # берём больше, чтобы после дедупа осталось top_k родителей
    raw = store.search(query, top_k=min(200, top_k * 10))

    best_by_parent: Dict[str, Dict[str, Any]] = {}
    for h in raw:
        pid = (h.get("parent_id") or "").strip()
        if not pid:
            continue
        score = float(h.get("score") or 0.0)
        pl = (h.get("payload") or {})

        prev = best_by_parent.get(pid)
        if (prev is None) or (score > float(prev.get("score", 0.0))):
            best_by_parent[pid] = {
                "parent_id": pid,
                "score": score,
                "title": str(pl.get("title", "")),
                "anchor": str(pl.get("anchor", "")),
                "section_path": str(pl.get("section_path", "")),
                "part_no": int(pl.get("parent_part_no") or 1),
                "doc_id": str(pl.get("doc_id", "")),
            }

    # сортируем и режем
    hits = sorted(best_by_parent.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k]
    return hits


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    POST /search
    """
    # 1) Qdrant (основной)
    try:
        raw_hits = _qdrant_parent_search(req.query, req.top_k)

        parent_ids = [h["parent_id"] for h in raw_hits]
        contents = repo.get_parent_texts(parent_ids)  # parent_id -> content

        hits: List[SearchHit] = []
        for h in raw_hits:
            pid = h["parent_id"]
            hits.append(
                SearchHit(
                    parent_id=pid,
                    score=float(h.get("score", 0.0)),
                    title=str(h.get("title", "")),
                    anchor=str(h.get("anchor", "")),
                    section_path=str(h.get("section_path", "")),
                    part_no=int(h.get("part_no", 1)),
                    doc_id=str(h.get("doc_id", "")),
                    content=contents.get(pid, ""),
                )
            )

        return SearchResponse(query=req.query, hits=hits)

    except Exception as e:
        log.warning("[search] qdrant failed -> fallback postgres. err=%r", e)

    # 2) Fallback: Postgres ILIKE
    raw_hits = _fallback_postgres_search(req.query, req.top_k)
    return SearchResponse(query=req.query, hits=[SearchHit(**h) for h in raw_hits])
