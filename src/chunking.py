from __future__ import annotations

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """
    Simple character chunker (good enough for logs).
    Later we can switch to token-based chunking.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks