from typing import List
from .schemas import SearchResult

def web_search(query: str) -> List[SearchResult]:
    # MVP: mock results so you can build the loop first.
    # Later replace this with a real search API call.
    return [
        SearchResult(
            title="Example source A",
            url="https://example.com/a",
            snippet=f"Snippet about: {query}"
        ),
        SearchResult(
            title="Example source B",
            url="https://example.com/b",
            snippet=f"Another snippet about: {query}"
        ),
    ]
