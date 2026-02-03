from dataclasses import dataclass, field
from typing import List, Dict, Any
from .schemas import SearchResult

@dataclass
class Memory:
    question: str
    notes: List[str] = field(default_factory=list)
    sources: List[SearchResult] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)  # agent steps/logs

    def add_trace(self, kind: str, payload: Any):
        self.trace.append({"kind": kind, "payload": payload})
