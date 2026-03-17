from __future__ import annotations

from pathlib import Path
import numpy as np
import faiss
from sqlalchemy import select

from langchain_ollama import OllamaEmbeddings, ChatOllama

from .db import SessionLocal
from .models import TextChunk, Experiment


INDEX_PATH = Path("faiss.index")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"


def retrieve(query: str, top_k: int = 5) -> list[TextChunk]:
    if not INDEX_PATH.exists():
        raise SystemExit("faiss.index not found. Run: python -m src.build_faiss")

    index = faiss.read_index(str(INDEX_PATH))

    emb = OllamaEmbeddings(model=EMBED_MODEL)
    qvec = np.array([emb.embed_query(query)], dtype="float32")

    distances, ids = index.search(qvec, top_k)
    faiss_ids = [int(x) for x in ids[0] if int(x) != -1]

    db = SessionLocal()
    chunks = db.scalars(select(TextChunk).where(TextChunk.faiss_id.in_(faiss_ids))).all()
    chunk_by_id = {c.faiss_id: c for c in chunks}
    ordered = [chunk_by_id[fid] for fid in faiss_ids if fid in chunk_by_id]
    db.close()
    return ordered


def answer_question(query: str, top_k: int = 5) -> str:
    chunks = retrieve(query, top_k=top_k)

    db = SessionLocal()

    # Create a compact context block
    context_lines = []
    for c in chunks:
        exp = db.scalar(select(Experiment).where(Experiment.id == c.experiment_id))
        exp_key = exp.exp_key if exp else f"exp_id={c.experiment_id}"
        context_lines.append(f"[{exp_key} | chunk {c.chunk_index} | faiss_id={c.faiss_id}]\n{c.content}")

    db.close()

    context = "\n\n---\n\n".join(context_lines) if context_lines else "(no context retrieved)"

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)

    prompt = f"""You are a helpful assistant for a computer vision researcher's experiment tracker.

Answer the user's question using ONLY the provided context.

Rules:
- If the answer is directly stated, give it directly.
- If the answer requires a small inference from the provided context, make that inference.
- Do not say the context is insufficient if the answer can be reasonably inferred from the retrieved chunks.
- When answering metric questions, include the metric values explicitly.
- When comparing experiments, mention the relevant differences such as augmentation, model, dataset, or metrics.

If the answer truly cannot be determined from the context, say what is missing and suggest what to ingest (metrics.json, config.yaml, train.log, etc).

QUESTION:
{query}

CONTEXT:
{context}

Answer:"""

    msg = llm.invoke(prompt)
    return msg.content


def main():
    import sys

    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.rag_answer "your question here"')

    query = " ".join(sys.argv[1:])
    chunks = retrieve(query, top_k=5)

    print("\n=== QUESTION ===")
    print(query)

    print("\n=== RETRIEVED CHUNKS ===")
    for i, c in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"faiss_id={c.faiss_id} experiment_id={c.experiment_id} chunk_index={c.chunk_index}")
        print(c.content[:800])

    print("\n=== ANSWER ===")
    print(answer_question(query, top_k=5))


if __name__ == "__main__":
    main()