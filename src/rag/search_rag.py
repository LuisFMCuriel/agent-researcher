from __future__ import annotations

from pathlib import Path

import numpy as np
import faiss
from sqlalchemy import select

from langchain_ollama import OllamaEmbeddings

from ..data.db import SessionLocal
from ..data.models import TextChunk, Experiment

INDEX_PATH = Path("faiss.index")
EMBED_MODEL = "nomic-embed-text"


def main():
    if not INDEX_PATH.exists():
        raise SystemExit("faiss.index not found. Run: python -m src.build_faiss")

    query = "cosine learning rate schedule cityscapes"
    top_k = 5

    # Load index + embed query
    index = faiss.read_index(str(INDEX_PATH))
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    qvec = np.array([emb.embed_query(query)], dtype="float32")

    # Search
    distances, ids = index.search(qvec, top_k)
    faiss_ids = [int(x) for x in ids[0] if int(x) != -1]

    db = SessionLocal()
    chunks = db.scalars(select(TextChunk).where(TextChunk.faiss_id.in_(faiss_ids))).all()

    # Keep original FAISS order
    chunk_by_id = {c.faiss_id: c for c in chunks}

    print("\n=== QUERY ===")
    print(query)

    print("\n=== RESULTS ===")
    for rank, fid in enumerate(faiss_ids, start=1):
        c = chunk_by_id.get(fid)
        if not c:
            continue
        exp = db.scalar(select(Experiment).where(Experiment.id == c.experiment_id))
        print(f"\n#{rank}  faiss_id={fid}  exp={exp.exp_key if exp else c.experiment_id}")
        print(c.content[:500])

    db.close()


if __name__ == "__main__":
    main()