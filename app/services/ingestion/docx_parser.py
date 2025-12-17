from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from docx import Document as DocxDocument


@dataclass(frozen=True)
class ParsedDocx:
    text: str
    title: str
    doc_number: str
    doc_date: str


def parse_docx(path: str | Path) -> ParsedDocx:
    p = Path(path)
    doc = DocxDocument(str(p))

    # Текст
    parts: list[str] = []
    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if t:
            parts.append(t)
    text = "\n".join(parts).strip()

    # Метаданные (минимум: из файла/первой строки)
    title = ""
    if parts:
        # часто 1-я строка в нормативке бывает названием
        title = parts[0].strip()
        # если это тупо номер типа "СП 70..." — ок, тоже годится
    if not title:
        title = p.stem

    # Пока не парсим номер/дату — оставляем пусто (ты это улучшишь позже)
    doc_number = ""
    doc_date = ""

    return ParsedDocx(text=text, title=title, doc_number=doc_number, doc_date=doc_date)
