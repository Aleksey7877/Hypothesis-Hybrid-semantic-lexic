from __future__ import annotations

import hashlib
import logging
import os
import uuid
import time
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.db.models import ChildChunk
from app.services.embeddings.e5 import E5Embedder

logger = logging.getLogger("uvicorn.error")

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://aedev-qdrant:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "child_chunks")

# ждать ли завершения операции апсерта (true = надёжнее, false = быстрее/асинхроннее)
QDRANT_WAIT = os.getenv("QDRANT_WAIT", "1").strip().lower() in {"1", "true", "yes", "on"}

# dev-флаг: пересоздавать коллекцию при несовпадении размерности
RECREATE_ON_DIM_MISMATCH = os.getenv("QDRANT_RECREATE_ON_DIM_MISMATCH", "0") == "1"


def _to_uuid(stable_id: str) -> str:
    """
    Qdrant point id должен быть int или uuid.
    Делаем стабильный UUID из stable_id.
    """
    h = hashlib.md5(stable_id.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex=h))


def _batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _safe_payload_value(x: Any) -> Any:
    """
    Приводим payload к JSON-safe типам (без numpy.int64 и т.п.).
    """
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, uuid.UUID):
        return str(x)
    return x


def _vec_to_list(v: Any) -> List[float]:
    """
    Приводим вектор к list[float], чтобы qdrant-client корректно сериализовал numpy/torch.
    """
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [float(x) for x in v]


def _extract_dim(info: Any) -> Optional[int]:
    """
    Достаём размерность из get_collection().
    Поддерживаем single-vector и named-vectors.
    """
    try:
        vectors = info.config.params.vectors  # type: ignore[attr-defined]
    except Exception:
        return None

    # single-vector
    try:
        if hasattr(vectors, "size"):
            return int(vectors.size)
    except Exception:
        pass

    # named vectors: dict[str, VectorParams]
    try:
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()))
            if hasattr(first, "size"):
                return int(first.size)
    except Exception:
        pass

    return None


class QdrantStore:
    def __init__(
        self,
        embedder: E5Embedder,
        url: str = DEFAULT_QDRANT_URL,
        collection: str = DEFAULT_COLLECTION,
        timeout: float = 60.0,
        prefer_grpc: bool = True,
    ):
        self.embedder = embedder
        self.collection = collection

        # gRPC быстрее, но требует открытого порта 6334 внутри docker-сети (у qdrant он есть по умолчанию)
        self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc, timeout=timeout)

        logger.warning(
            "[qdrant] init url=%s collection=%s dim=%s wait=%s",
            url,
            collection,
            self.embedder.dim(),
            QDRANT_WAIT,
        )

    def ensure_collection(self) -> None:
        dim = int(self.embedder.dim())

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing_dim = _extract_dim(info)
            if existing_dim is not None and existing_dim != dim:
                msg = f"Qdrant collection '{self.collection}' dim={existing_dim} but embedder dim={dim}"
                if RECREATE_ON_DIM_MISMATCH:
                    logger.warning("[qdrant] %s -> recreating (dev).", msg)
                    self.client.delete_collection(self.collection)
                else:
                    raise ValueError(msg)
            else:
                return

        logger.warning("[qdrant] create_collection name=%s dim=%s", self.collection, dim)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            on_disk_payload=True,
        )

    def upsert_children(self, children: List[ChildChunk], batch_size: int = 512) -> int:
        """
        Батчами:
          1) эмбеддим тексты
          2) апсертим в Qdrant
        Возвращаем число отправленных points.
        """
        if not children:
            return 0

        self.ensure_collection()

        total = 0
        logger.warning(
            "[qdrant] upsert start collection=%s children=%d batch=%d",
            self.collection,
            len(children),
            batch_size,
        )

        for bi, batch in enumerate(_batches(children, batch_size), start=1):
            t0 = time.perf_counter()
            texts = [c.content for c in batch]
            vectors = self.embedder.encode_passages(texts)
            logger.warning("[qdrant] embed done batch=%d size=%d (%.2fs)", bi, len(batch), time.perf_counter() - t0)

            points: List[qm.PointStruct] = []
            for c, v in zip(batch, vectors):
                payload: Dict[str, Any] = {
                    "child_id": _safe_payload_value(c.child_id),
                    "parent_id": _safe_payload_value(c.parent_id),
                    "doc_id": _safe_payload_value(c.doc_id),
                    "child_no": _safe_payload_value(c.child_no),
                    "start_char": _safe_payload_value(c.start_char),
                    "end_char": _safe_payload_value(c.end_char),
                    "section_path": _safe_payload_value(c.section_path),
                    "anchor": _safe_payload_value(c.anchor),
                    "title": _safe_payload_value(c.title),
                    "parent_part_no": _safe_payload_value(c.parent_part_no),
                    "content": _safe_payload_value(c.content),
                }
                points.append(
                    qm.PointStruct(
                        id=_to_uuid(str(c.child_id)),
                        vector=_vec_to_list(v),
                        payload=payload,
                    )
                )

            t1 = time.perf_counter()
            self.client.upsert(collection_name=self.collection, points=points, wait=QDRANT_WAIT)
            total += len(points)

            logger.warning(
                "[qdrant] upsert done batch=%d points=%d (%.2fs) total=%d",
                bi,
                len(points),
                time.perf_counter() - t1,
                total,
            )

        logger.warning("[qdrant] upsert finished total=%d", total)
        return total

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Векторный поиск по Qdrant (по child_chunks).
        Возвращает список raw-хитов (по чанкам), где в payload лежит parent_id и метаданные.
        """
        self.ensure_collection()

        qvec = _vec_to_list(self.embedder.encode_query(query))

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        out: List[Dict[str, Any]] = []
        for h in hits:
            pl = h.payload or {}
            out.append(
                {
                    "parent_id": str(pl.get("parent_id", "")),
                    "score": float(h.score or 0.0),
                    "payload": pl,
                }
            )
        return out
