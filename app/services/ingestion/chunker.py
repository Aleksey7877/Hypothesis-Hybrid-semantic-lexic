from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

from app.db.models import ParentChunk, ChildChunk
from app.services.ingestion.structure_parser import split_to_blocks, TextBlock
from app.services.chunking.child_chunker import make_children, ChildChunkConfig


@dataclass
class ParentChunkConfig:
    max_len: int = 2500
    overlap: int = 200
    min_len: int = 400


def _stable_parent_id(doc_id: str, part_no: int, content: str) -> str:
    h = hashlib.sha1()
    h.update(doc_id.encode("utf-8"))
    h.update(b":")
    h.update(str(part_no).encode("utf-8"))
    h.update(b":")
    h.update(content.encode("utf-8", errors="ignore"))
    return "parent_" + h.hexdigest()[:40]


def _smart_cut(text: str, start: int, max_end: int) -> int:
    window = text[start:max_end]
    for sep in ["\n\n", "\n", ". ", " "]:
        pos = window.rfind(sep)
        if pos != -1 and pos > int(len(window) * 0.55):
            return start + pos + len(sep)
    return max_end


def make_parents_from_text(
    *,
    doc_id: str,
    doc_title: str,
    text: str,
    cfg: ParentChunkConfig = ParentChunkConfig(),
) -> List[ParentChunk]:
    """
    Parent чанкинг с попыткой “держать структуру”:
    - сначала режем на блоки по якорям
    - затем упаковываем блоки в куски ~max_len
    Если якорей нет — режем по длине smart_cut.
    """
    blocks = split_to_blocks(text)

    # если структура слабая — проще резать по длине
    if len(blocks) <= 1:
        t = (text or "").strip()
        if not t:
            return []
        out: list[ParentChunk] = []
        n = len(t)
        i = 0
        part_no = 1
        while i < n:
            max_end = min(i + cfg.max_len, n)
            end = _smart_cut(t, i, max_end)
            chunk = t[i:end].strip()
            if len(chunk) >= cfg.min_len:
                out.append(
                    ParentChunk(
                        parent_id=_stable_parent_id(doc_id, part_no, chunk),
                        doc_id=doc_id,
                        section_path="",
                        anchor="",
                        title=doc_title or "",
                        part_no=part_no,
                        content=chunk,
                    )
                )
                part_no += 1
            if end >= n:
                break
            i = max(0, end - cfg.overlap)
        return out

    # упаковка блоков
    out: list[ParentChunk] = []
    buf: list[str] = []
    meta_anchor = ""
    meta_path = ""
    meta_title = doc_title or ""
    part_no = 1

    def flush():
        nonlocal buf, part_no, meta_anchor, meta_path, meta_title
        content = "\n".join([x for x in buf if x.strip()]).strip()
        if len(content) >= cfg.min_len:
            out.append(
                ParentChunk(
                    parent_id=_stable_parent_id(doc_id, part_no, content),
                    doc_id=doc_id,
                    section_path=meta_path or "",
                    anchor=meta_anchor or "",
                    title=meta_title or "",
                    part_no=part_no,
                    content=content,
                )
            )
            part_no += 1
        buf = []
        meta_anchor = ""
        meta_path = ""
        meta_title = doc_title or ""

    for b in blocks:
        header = ""
        if b.anchor or b.heading:
            header = f"{b.anchor} {b.heading}".strip()
        piece = "\n".join([x for x in [header, b.text] if x]).strip()

        if not meta_anchor and b.anchor:
            meta_anchor = b.anchor
        if not meta_path and b.section_path:
            meta_path = b.section_path
        if meta_title == (doc_title or "") and b.heading:
            meta_title = b.heading

        candidate = ("\n".join(buf + [piece])).strip()
        if len(candidate) > cfg.max_len and buf:
            flush()
            # стартуем новый parent с текущего блока
            if b.anchor:
                meta_anchor = b.anchor
            meta_path = b.section_path
            meta_title = b.heading or (doc_title or "")
            buf.append(piece)
        else:
            buf.append(piece)

    flush()
    return out


def chunk_document(
    *,
    doc_id: str,
    doc_title: str,
    text: str,
    parent_cfg: ParentChunkConfig = ParentChunkConfig(),
    child_cfg: ChildChunkConfig = ChildChunkConfig(),
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    parents = make_parents_from_text(doc_id=doc_id, doc_title=doc_title, text=text, cfg=parent_cfg)
    children: list[ChildChunk] = []
    for p in parents:
        children.extend(make_children(p, child_cfg))
    return parents, children
