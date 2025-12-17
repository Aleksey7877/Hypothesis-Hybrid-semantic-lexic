import re

_ws_re = re.compile(r"\s+")

def normalize(text: str) -> str:
    return _ws_re.sub(" ", (text or "").strip())

def split_paragraphs(text: str) -> list[str]:
    # keep non-empty lines
    return [normalize(t) for t in re.split(r"\n+", text) if normalize(t)]
