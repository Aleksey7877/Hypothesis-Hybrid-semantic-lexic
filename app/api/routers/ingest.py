from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.db.models import ParentChunk
from app.db.repository import Repository
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.chunking.child_chunker import ChildChunkConfig, make_children
from app.services.embeddings.e5 import E5Embedder
from app.services.vectorstore.qdrant import QdrantStore
from app.services.lexical.elastic import ElasticStore

log = logging.getLogger(__name__)
router = APIRouter()


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _stable_id(prefix: str, b: bytes) -> str:
    h = hashlib.sha1(b).hexdigest()
    return f"{prefix}_{h}"


def _guess_doc_meta(text: str, fallback_title: str) -> Tuple[str, str, str]:
    title = fallback_title
    doc_number = ""
    doc_date = ""

    m = re.search(r"(№|N)\s*([0-9A-Za-z\-/]+)", text)
    if m:
        doc_number = m.group(2)

    d = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if d:
        doc_date = d.group(1)

    # Title: первая непустая строка
    for line in text.splitlines():
        line = line.strip()
        if line:
            title = line[:160]
            break

    return title, doc_number, doc_date


def _extract_text_from_docx(fp: Path) -> str:
    # аккуратно, чтобы не утонуть в таблицах/пустых параграфах
    from docx import Document  # type: ignore

    doc = Document(str(fp))
    parts: List[str] = []
    last = ""

    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t and t != last:
            parts.append(t)
            last = t

    # таблицы тоже полезны
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                t = (cell.text or "").strip()
                if t and t != last:
                    parts.append(t)
                    last = t

    return "\n".join(parts).strip()


def _parent_chunk(text: str, parent_chars: int) -> List[Tuple[int, str]]:
    if not text:
        return []
    out: List[Tuple[int, str]] = []
    part_no = 1
    start = 0
    n = len(text)
    while start < n:
        chunk = text[start : start + parent_chars].strip()
        if chunk:
            out.append((part_no, chunk))
            part_no += 1
        start += parent_chars
    return out


# singleton store
_qdrant_store: QdrantStore | None = None


def get_qdrant_store() -> QdrantStore:
    global _qdrant_store
    if _qdrant_store is None:
        embedder = E5Embedder()
        _qdrant_store = QdrantStore(embedder=embedder)
    return _qdrant_store


_elastic_store: ElasticStore | None = None


def get_elastic_store() -> ElasticStore:
    global _elastic_store
    if _elastic_store is None:
        _elastic_store = ElasticStore()
    return _elastic_store


def _build_children(parents: List[ParentChunk], child_cfg: ChildChunkConfig) -> List:
    """
    Строим children. Опционально в несколько потоков через CHILD_WORKERS.
    """
    workers = int(os.getenv("CHILD_WORKERS", "1"))
    workers = max(1, min(workers, 32))

    log.warning("[ingest] child_workers=%d", workers)

    if workers == 1:
        children = []
        for i, p in enumerate(parents, start=1):
            children.extend(make_children(p, child_cfg))
            if i % 25 == 0:
                log.warning(
                    "[ingest] children_build_progress parents=%d/%d children=%d",
                    i, len(parents), len(children)
                )
        return children

    children = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(make_children, p, child_cfg) for p in parents]
        done = 0
        for f in as_completed(futures):
            children.extend(f.result())
            done += 1
            if done % 25 == 0:
                log.warning(
                    "[ingest] children_build_progress parents=%d/%d children=%d",
                    done, len(parents), len(children)
                )
    return children


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    t0 = time.perf_counter()

    repo = Repository()

    parent_chars = int(getattr(settings, "PARENT_CHARS", 2500))
    enable_indexing = _truthy(os.getenv("ENABLE_INDEXING", "0"))
    enable_elastic_indexing = _truthy(os.getenv("ENABLE_ELASTIC_INDEXING", os.getenv("ENABLE_INDEXING", "0")))

    folder = Path(req.folder_path)
    if not folder.is_absolute():
        folder = Path("/app") / folder

    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"folder_path not found: {folder}")

    exts = {e.lower() for e in req.extensions}
    paths = list(folder.rglob("*")) if req.recursive else list(folder.glob("*"))
    files = [
        p
        for p in paths
        if p.is_file()
        and p.suffix.lower() in exts
        and not p.name.startswith("~$")
    ]

    log.warning(
        "[ingest] start folder=%s files=%d parent_chars=%d enable_indexing=%s enable_elastic_indexing=%s",
        str(folder), len(files), parent_chars, enable_indexing, enable_elastic_indexing
    )

    indexed_files = 0
    parents_total = 0
    children_db_total = 0
    skipped_files = 0
    errors: List[str] = []

    qdrant_store: QdrantStore | None = get_qdrant_store() if enable_indexing else None

    elastic_store: ElasticStore | None = get_elastic_store() if enable_elastic_indexing else None
    if elastic_store is not None:
        elastic_store.ensure_index()

    # child chunk config
    child_cfg = ChildChunkConfig(max_len=600, overlap=80, min_len=120)
    qdrant_batch = int(os.getenv("QDRANT_BATCH", "512"))
    log.warning("[ingest] qdrant_batch=%d", qdrant_batch)

    for fp in files:
        file_t0 = time.perf_counter()
        try:
            # пока поддерживаем docx (rtf у тебя раньше отваливался импортом)
            if fp.suffix.lower() != ".docx":
                skipped_files += 1
                continue

            log.warning("[ingest] file=%s", fp.name)

            text = _extract_text_from_docx(fp)
            if not text:
                skipped_files += 1
                log.warning("[ingest] skip empty text=%s", fp.name)
                continue

            log.warning("[ingest] text_len=%d", len(text))

            blob = fp.read_bytes()
            doc_id = _stable_id("doc", blob)
            title, doc_number, doc_date = _guess_doc_meta(text, fallback_title=fp.name)

            # clean old doc rows (parents/children too, если так реализовано в repo)
            repo.delete_document(doc_id)
            repo.upsert_document(doc_id=doc_id, title=title, doc_number=doc_number, doc_date=doc_date)

            parents: List[ParentChunk] = []
            for part_no, chunk_text in _parent_chunk(text, parent_chars):
                parent_payload = f"{doc_id}:{part_no}:{chunk_text[:200]}".encode("utf-8", errors="ignore")
                parent_id = _stable_id("parent", parent_payload)
                parents.append(
                    ParentChunk(
                        parent_id=parent_id,
                        doc_id=doc_id,
                        section_path="",
                        anchor="",
                        title=title,
                        part_no=part_no,
                        content=chunk_text,
                    )
                )

            pw = repo.upsert_parents(parents)
            parents_total += pw
            log.warning("[ingest] parents_written=%d parents_in_mem=%d", pw, len(parents))

            # 0) index parents to Elasticsearch (optional)
            if enable_elastic_indexing and parents:
                if elastic_store is None:
                    raise RuntimeError("ENABLE_ELASTIC_INDEXING is true but elastic_store is not initialized")

                t_es0 = time.perf_counter()
                try:
                    elastic_store.delete_by_doc_id(doc_id)
                    ew = elastic_store.bulk_index_parents(parents, refresh=False)
                    log.warning(
                        "[ingest] elastic_written=%d (%.2fs) index=%s",
                        ew, time.perf_counter() - t_es0, elastic_store.index
                    )
                except Exception as e:
                    log.warning("[ingest] elastic_failed=%s err=%r", fp.name, e)

            # build children
            t_children0 = time.perf_counter()
            log.warning("[ingest] children_build_start")

            children = _build_children(parents, child_cfg)

            log.warning(
                "[ingest] children_build_done children=%d (%.2fs)",
                len(children), time.perf_counter() - t_children0
            )

            # 1) write children to Postgres
            t_db0 = time.perf_counter()
            cw = repo.upsert_children(children)
            children_db_total += cw
            log.warning("[ingest] children_db_written=%d (%.2fs)", cw, time.perf_counter() - t_db0)

            # 2) upsert to Qdrant (optional)
            if enable_indexing and children:
                if qdrant_store is None:
                    raise RuntimeError("ENABLE_INDEXING is true but qdrant_store is not initialized")

                t_up0 = time.perf_counter()
                qw = qdrant_store.upsert_children(children, batch_size=qdrant_batch)
                log.warning(
                    "[ingest] qdrant_written=%d (%.2fs) collection=%s",
                    qw, time.perf_counter() - t_up0, qdrant_store.collection
                )

            indexed_files += 1
            log.warning("[ingest] file_done=%s (%.2fs)", fp.name, time.perf_counter() - file_t0)

        except Exception as e:
            log.warning("[ingest] file_failed=%s err=%r", fp.name, e)
            errors.append(f"{fp.name}: {type(e).__name__}: {e}")

    log.warning(
        "[ingest] done indexed=%d parents=%d children_db=%d skipped=%d (%.2fs)",
        indexed_files, parents_total, children_db_total, skipped_files, time.perf_counter() - t0
    )

    return IngestResponse(
        indexed_files=indexed_files,
        parents=parents_total,
        children=children_db_total,
        skipped_files=skipped_files,
        errors=errors,
    )
