from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


ANCHOR_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,8})[)\.\s]+(.*)$")


@dataclass
class TextBlock:
    anchor: str
    heading: str
    section_path: str
    text: str


def _build_section_path(stack: list[str]) -> str:
    # "1 > 1.2 > 1.2.3"
    return " > ".join(stack)


def split_to_blocks(text: str) -> List[TextBlock]:
    """
    Очень простой парсер структуры:
    - ищет строки вида "1.2.3. Заголовок" / "1.2) Заголовок"
    - всё остальное — в текущий блок
    """
    lines = (text or "").splitlines()

    blocks: list[TextBlock] = []
    cur_anchor = ""
    cur_heading = ""
    cur_stack: list[str] = []
    buf: list[str] = []

    def flush():
        nonlocal buf, cur_anchor, cur_heading
        t = "\n".join(buf).strip()
        if not t and not cur_anchor and not cur_heading:
            buf = []
            return
        blocks.append(
            TextBlock(
                anchor=cur_anchor,
                heading=cur_heading,
                section_path=_build_section_path(cur_stack),
                text=t,
            )
        )
        buf = []

    for line in lines:
        s = (line or "").rstrip()
        m = ANCHOR_RE.match(s)
        if m:
            # новый якорь — закрываем прошлый блок
            flush()
            cur_anchor = m.group(1).strip()
            cur_heading = (m.group(2) or "").strip()

            # обновляем стек "1.2.3" => ["1", "1.2", "1.2.3"]
            parts = cur_anchor.split(".")
            cur_stack = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
        else:
            buf.append(s)

    flush()
    # если вообще нет якорей — один блок “без структуры”
    if not blocks:
        blocks = [TextBlock(anchor="", heading="", section_path="", text=(text or "").strip())]
    return blocks
