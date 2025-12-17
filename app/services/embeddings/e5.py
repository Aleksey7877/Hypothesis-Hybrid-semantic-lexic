from __future__ import annotations

import os
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


DEFAULT_MODEL = os.getenv("E5_MODEL_NAME", "intfloat/multilingual-e5-large")
DEFAULT_BATCH = int(os.getenv("E5_BATCH_SIZE", "64"))
DEFAULT_DEVICE = os.getenv("E5_DEVICE", "auto")  # auto|cpu|cuda


def _pick_device(dev: str) -> str:
    dev = (dev or "auto").strip().lower()
    if dev in {"cpu", "cuda"}:
        return dev
    # auto
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class E5Embedder:
    """
    E5 любит префиксы:
      - query: ...
      - passage: ...
    """

    def __init__(self, model_name: str | None = None, batch_size: int | None = None, device: str | None = None):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")

        self.model_name = model_name or DEFAULT_MODEL
        self.batch_size = int(batch_size or DEFAULT_BATCH)
        self.device = _pick_device(device or DEFAULT_DEVICE)

        # ВАЖНО: грузим сразу на device
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def dim(self) -> int:
        return self._dim

    def encode_query(self, query: str) -> List[float]:
        vec = self._model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32).tolist()

    def encode_passages(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.astype(np.float32).tolist() for v in vecs]
