import hashlib


def stable_id(s: str) -> str:
    # стабильный короткий id (удобно для parent_id/child_id)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:16]
