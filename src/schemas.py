from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: Optional[float] = None
