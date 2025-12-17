from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

from app.db.models import ParentChunk, ChildChunk


# --- структурные маркеры (сильные границы) ---
STRUCT_MARKERS = [
    "\n\nГлава ", "\n\nГЛАВА ",
    "\n\nРаздел ", "\n\nРАЗДЕЛ ",
    "\n\nСтатья ", "\n\nСТАТЬЯ ",
    "\n\nПриложение", "\n\nПРИЛОЖЕНИЕ",

    "\n\nОбласть применения",
    "\n\nНормативные ссылки",
    "\n\nТермины и определения",
    "\n\nОбщие положения",
    "\n\nВведение",
    "\n\nПредисловие",
    "\n\nСодержание",

    "\n\nТаблица ", "\n\nРисунок ", "\n\nРис. ", "\n\nТабл. ",
    "\n\nПримечание", "\n\nПРИМЕЧАНИЕ",
]

# числовые заголовки: "\n 4.2.1 Заголовок" (рус/лат заглавные)
_HEADING_RE = re.compile(r"\n\s*\d{1,3}(\.\d{1,3}){0,6}\s+[А-ЯЁA-Z]", re.MULTILINE)


def _find_struct_boundary(window: str) -> Optional[int]:
    """
    Возвращает позицию (offset в window) последней "сильной" структурной границы.
    """
    best: Optional[int] = None

    # 1) последние вхождения по явным маркерам
    for m in STRUCT_MARKERS:
        p = window.rfind(m)
        if p != -1:
            best = p if (best is None or p > best) else best

    # 2) последний “числовой заголовок”
    last: Optional[int] = None
    for mm in _HEADING_RE.finditer(window):
        last = mm.start()
    if last is not None:
        best = last if (best is None or last > best) else best

    return best


@dataclass
class ChildChunkConfig:
    max_len: int = 600
    overlap: int = 80
    min_len: int = 80


def _stable_id(parent_id: str, child_no: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(parent_id.encode("utf-8"))
    h.update(b":")
    h.update(str(child_no).encode("utf-8"))
    h.update(b":")
    h.update(text.encode("utf-8", errors="ignore"))
    return "child_" + h.hexdigest()[:40]


def _smart_cut(text: str, start: int, max_end: int, max_len: int) -> int:
    """
    Возвращает end (int) в диапазоне [start+1 .. max_end].
    Сначала пытаемся резать по структурным границам, иначе по \n\n, затем \n, затем пробел.
    """
    window = text[start:max_end]

    # 0) структурные границы (если нашли ближе к концу окна)
    b = _find_struct_boundary(window)
    if b is not None:
        # не режем слишком рано: пусть хотя бы 30% чанка наберётся
        min_cut = max(80, int(max_len * 0.30))
        if b >= min_cut:
            return start + b

    # 1) двойной перенос
    p = window.rfind("\n\n")
    if p != -1 and p >= 120:
        return start + p

    # 2) один перенос
    p = window.rfind("\n")
    if p != -1 and p >= 120:
        return start + p

    # 3) пробел
    p = window.rfind(" ")
    if p != -1 and p >= 120:
        return start + p

    # 4) fallback
    return max_end


def make_children(parent: ParentChunk, cfg: ChildChunkConfig = ChildChunkConfig()) -> List[ChildChunk]:
    t = (parent.content or "").strip()
    if not t:
        return []

    # защита от некорректной конфигурации
    max_len = max(1, int(cfg.max_len))
    overlap = max(0, int(cfg.overlap))
    if overlap >= max_len:
        overlap = max_len // 3

    out: List[ChildChunk] = []
    n = len(t)
    i = 0
    child_no = 1

    # предохранитель от вечных циклов
    max_iters = max(10_000, (n // max_len + 1) * 50)
    iters = 0

    while i < n:
        iters += 1
        if iters > max_iters:
            break

        max_end = min(i + max_len, n)
        end = _smart_cut(t, i, max_end, max_len)

        # гарантируем прогресс
        if end <= i:
            end = max_end
        if end <= i:
            end = min(i + 1, n)

        chunk = t[i:end].strip()

        if len(chunk) >= int(cfg.min_len):
            out.append(
                ChildChunk(
                    child_id=_stable_id(parent.parent_id, child_no, chunk),
                    parent_id=parent.parent_id,
                    doc_id=parent.doc_id,
                    child_no=child_no,
                    start_char=i,
                    end_char=end,
                    section_path=parent.section_path,
                    anchor=parent.anchor,
                    title=parent.title,
                    parent_part_no=parent.part_no,
                    content=chunk,
                )
            )
            child_no += 1

        if end >= n:
            break

        next_i = end - overlap
        if next_i <= i:
            next_i = end
        i = next_i

    return out
