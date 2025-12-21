from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger("uvicorn.error")


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class BGERerankConfig:
    model_name: str
    cache_dir: str
    device: str           # "cpu" | "cuda" | "auto"
    batch_size: int
    max_len: int
    fp16: bool
    enabled: bool
    normalize: bool       # <= NEW


_lock = threading.Lock()
_ready = False
_cached: Tuple[Any, Any, BGERerankConfig, torch.device] | None = None


def get_cfg() -> BGERerankConfig:
    model_name = os.getenv("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3").strip()
    cache_dir = os.getenv("BGE_RERANK_CACHE_DIR", "/app/data/hf_cache").strip()

    device = os.getenv("BGE_RERANK_DEVICE", "cpu").strip().lower()
    batch_size = int(os.getenv("BGE_RERANK_BATCH", "8"))
    max_len = int(os.getenv("BGE_RERANK_MAX_LEN", "512"))
    fp16 = _truthy(os.getenv("BGE_RERANK_FP16", "0"))
    enabled = _truthy(os.getenv("BGE_RERANK_ENABLED", "1"))

    # normalize=True как в примере FlagEmbedding/HF
    normalize = _truthy(os.getenv("BGE_RERANK_NORMALIZE", "1"))

    # safety clamps
    batch_size = max(1, min(batch_size, 64))
    max_len = max(32, min(max_len, 2048))

    return BGERerankConfig(
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
        fp16=fp16,
        enabled=enabled,
        normalize=normalize,
    )


def is_ready() -> bool:
    return _ready


def _resolve_device(cfg: BGERerankConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_hf_env(cache_dir: str) -> None:
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_XET", os.getenv("HF_HUB_DISABLE_XET", "1"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(cache_dir, exist_ok=True)


def _load_model() -> Tuple[Any, Any, BGERerankConfig, torch.device]:
    global _cached, _ready

    cfg = get_cfg()
    if not cfg.enabled:
        _ready = False
        raise RuntimeError("BGE reranker disabled via BGE_RERANK_ENABLED=0")

    _ensure_hf_env(cfg.cache_dir)

    if _cached is not None:
        return _cached

    with _lock:
        if _cached is not None:
            return _cached

        dev = _resolve_device(cfg)
        dtype = torch.float16 if (cfg.fp16 and dev.type == "cuda") else torch.float32

        log.warning(
            "[bge] loading model=%s device=%s fp16=%s cache=%s normalize=%s",
            cfg.model_name, dev.type, cfg.fp16, cfg.cache_dir, cfg.normalize
        )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            use_fast=True,
            trust_remote_code=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            torch_dtype=dtype,
            trust_remote_code=False,
        )
        model.to(dev)
        model.eval()

        _cached = (tokenizer, model, cfg, dev)
        _ready = True
        return _cached


def warmup() -> None:
    tokenizer, model, cfg, dev = _load_model()

    pair = tokenizer(
        ["warmup query"],
        ["warmup passage"],
        padding=True,
        truncation=True,
        max_length=cfg.max_len,
        return_tensors="pt",
    )
    pair = {k: v.to(dev) for k, v in pair.items()}

    with torch.inference_mode():
        _ = model(**pair).logits


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Приводим score к [0..1] примерно как normalize=True в FlagEmbedding.
    """
    # logits: [B,1] или [B,2] (редко)
    if logits.ndim == 2 and logits.size(-1) == 1:
        return torch.sigmoid(logits).squeeze(-1)  # [B]
    if logits.ndim == 2 and logits.size(-1) == 2:
        probs = torch.softmax(logits, dim=-1)      # [B,2]
        return probs[:, 1]                         # [B]
    # fallback: просто squeeze
    return logits.squeeze(-1)


def rerank_passages(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    passages: [{parent_id, text, base_rank, base_score, source, ...}]
    returns:  same dicts + rerank_score, sorted desc by rerank_score
    """
    if not passages:
        return []

    cfg = get_cfg()
    if not cfg.enabled:
        out = []
        for p in passages:
            q = dict(p)
            q["rerank_score"] = 0.0
            out.append(q)
        return out

    tokenizer, model, cfg, dev = _load_model()

    texts = [str(p.get("text") or "") for p in passages]
    n = len(texts)

    scores: List[float] = []
    bs = cfg.batch_size

    with torch.inference_mode():
        for i in range(0, n, bs):
            batch_texts = texts[i : i + bs]
            batch_queries = [query] * len(batch_texts)

            enc = tokenizer(
                batch_queries,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=cfg.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc).logits

            if cfg.normalize:
                batch = _normalize_logits(logits).detach().float().cpu().tolist()
            else:
                batch = logits.squeeze(-1).detach().float().cpu().tolist()

            if isinstance(batch, float):
                batch = [batch]
            scores.extend([float(x) for x in batch])

    out: List[Dict[str, Any]] = []
    for p, s in zip(passages, scores):
        q = dict(p)
        q["rerank_score"] = float(s)
        out.append(q)

    out.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    return out
