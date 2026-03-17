from typing import Dict, Any
from .memory import Memory
from .tools import web_search

def decide_next_action(mem: Memory) -> Dict[str, Any]:
    """
    MVP policy:
    - If no sources yet: search using the question.
    - Else: finish with a simple synthesized answer.
    Later: replace with an LLM planner.
    """
    if not mem.sources:
        return {"tool": "web_search", "args": {"query": mem.question}}

    citations = [{"title": s.title, "url": s.url} for s in mem.sources]
    answer = (
        f"Draft answer to: {mem.question}\n\n"
        f"I found {len(mem.sources)} sources and synthesized the snippets.\n"
        f"(Replace this with LLM synthesis in the next step.)"
    )
    return {"tool": "finish", "args": {"answer": answer, "citations": citations}}

def run_agent(question: str, max_steps: int = 5) -> Dict[str, Any]:
    mem = Memory(question=question)

    for step in range(max_steps):
        action = decide_next_action(mem)
        mem.add_trace("action", action)

        tool = action["tool"]
        args = action["args"]

        if tool == "web_search":
            results = web_search(args["query"])
            mem.sources.extend(results)
            mem.add_trace("observation", {"results": [r.__dict__ for r in results]})

        elif tool == "finish":
            mem.add_trace("finish", args)
            return {
                "answer": args["answer"],
                "citations": args["citations"],
                "trace": mem.trace,
            }

        else:
            mem.add_trace("error", f"Unknown tool: {tool}")
            break

    return {
        "answer": "Stopped due to max_steps (no finish called).",
        "citations": [],
        "trace": mem.trace,
    }
