from __future__ import annotations

from pathlib import Path
import os
import sys
import faiss
from sqlalchemy import select, func

from langchain_ollama import OllamaEmbeddings
import numpy as np
from .db import SessionLocal
from .models import ExperimentArtifact, TextChunk
from .chunking import chunk_text


INDEX_PATH = Path("faiss.index")
EMBED_MODEL = "nomic-embed-text"
VECTOR_DIM = 768  # nomic-embed-text returned 768 in your test


def load_or_create_index() -> faiss.Index:
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))

    base = faiss.IndexFlatL2(VECTOR_DIM)
    return faiss.IndexIDMap2(base)


def main():
    import sys

    db = SessionLocal()  # <-- create db session FIRST

    rebuild = "--rebuild" in sys.argv
    if rebuild:
        # delete SQL chunks
        db.query(TextChunk).delete()
        db.commit()

        # delete index file if exists
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()

    # Load (or create) FAISS index AFTER optional deletion
    index = load_or_create_index()

    # Determine next faiss_id
    next_faiss_id = index.ntotal

    # Embeddings
    emb = OllamaEmbeddings(model=EMBED_MODEL)

    artifacts = db.scalars(select(ExperimentArtifact)).all()
    if not artifacts:
        print("No artifacts found. Ingest an experiment first.")
        db.close()
        return

    added = 0
    for art in artifacts:
        chunks = chunk_text(art.content, chunk_size=900, overlap=150)
        if not chunks:
            continue

        vectors = emb.embed_documents(chunks)

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            faiss_id = next_faiss_id
            next_faiss_id += 1

            vec_np = np.array([vec], dtype="float32")
            id_np = np.array([faiss_id], dtype="int64")

            index.add_with_ids(vec_np, id_np)

            db.add(
                TextChunk(
                    experiment_id=art.experiment_id,
                    artifact_id=art.id,
                    chunk_index=i,
                    content=chunk,
                    faiss_id=faiss_id,
                )
            )
            added += 1

    db.commit()
    db.close()

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Indexed {added} chunks into {INDEX_PATH} (total vectors={index.ntotal}).")


if __name__ == "__main__":
    main()