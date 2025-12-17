\
import re
from dataclasses import dataclass
from typing import Optional

# Top-level section: "1 Область применения"
RE_SECTION = re.compile(r"^\s*(\d+)\s+(.+)$")

# Numeric clause: "7.8.1 Текст..."
RE_CLAUSE_NUM = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.+)$")

# Appendix header: "Приложение С (обязательное)"
RE_APPENDIX_HDR = re.compile(r"^\s*Приложение\s+([A-ZА-Я])\b", re.IGNORECASE)

# Appendix clause: "С.1.2 Текст..."
RE_CLAUSE_APP = re.compile(r"^\s*([A-ZА-Я])\.(\d+(?:\.\d+)*)\s+(.+)$")

# Query anchor patterns (find inside)
RE_ANCHOR_IN_QUERY = re.compile(r"(?<!\d)(\d+(?:\.\d+)+)")
RE_ANCHOR_APP_IN_QUERY = re.compile(r"\b([A-ZА-Я])\.\d+(?:\.\d+)*\b")

@dataclass(frozen=True)
class AnchorMatch:
    anchor: str
    title: str
    kind: str  # section|clause|appendix|appendix_clause

def detect_anchor(line: str, current_appendix: Optional[str] = None) -> Optional[AnchorMatch]:
    line = (line or "").strip()
    if not line:
        return None

    # Appendix header
    m = RE_APPENDIX_HDR.match(line)
    if m:
        letter = m.group(1).upper()
        return AnchorMatch(anchor=letter, title=line, kind="appendix")

    # Appendix clause (letter-based)
    m = RE_CLAUSE_APP.match(line)
    if m:
        letter = m.group(1).upper()
        nums = m.group(2)
        title = m.group(3).strip()
        return AnchorMatch(anchor=f"{letter}.{nums}", title=title, kind="appendix_clause")

    # Numeric clause
    m = RE_CLAUSE_NUM.match(line)
    if m:
        return AnchorMatch(anchor=m.group(1), title=m.group(2).strip(), kind="clause")

    # Top-level section: must NOT be "1.1 ..."
    m = RE_SECTION.match(line)
    if m and "." not in m.group(1):
        # guard: "1.1" will not match because group1 is digits only
        return AnchorMatch(anchor=m.group(1), title=m.group(2).strip(), kind="section")

    return None

def extract_query_anchors(query: str) -> list[str]:
    q = query or ""
    anchors = set(RE_ANCHOR_IN_QUERY.findall(q))
    anchors.update(RE_ANCHOR_APP_IN_QUERY.findall(q))  # returns letter only; expand below
    # Expand appendix anchors found as full tokens
    for tok in RE_ANCHOR_APP_IN_QUERY.findall(q):
        # We only got letter from capturing group; but full match exists too.
        pass
    # Better: find full appendix anchors
    for m in re.finditer(r"\b([A-ZА-Я]\.\d+(?:\.\d+)*)\b", q):
        anchors.add(m.group(1))
    return sorted(anchors)
