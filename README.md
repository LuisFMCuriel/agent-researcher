# Agent Researcher – RAG System

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for analyzing and querying machine learning experiment results.

The system combines:
- A **SQL database** for structured and unstructured data
- A **FAISS vector index** for similarity search
- A **local LLM (via Ollama)** for generating grounded answers

The goal is to enable users to ask natural language questions about experiments and receive answers based on stored experiment data.

---

## Architecture

The system consists of three main layers:

### 1. Data Layer (SQL)

Stores experiment data in three tables:

- **experiments**
  - Structured data (metrics, model, dataset, notes)

- **experiment_artifacts**
  - Full textual reports for each experiment

- **text_chunks**
  - Chunked pieces of artifact text
  - Each chunk is linked to a FAISS vector via `faiss_id`

---

### 2. Retrieval Layer (FAISS)

- Stores vector embeddings of text chunks
- Enables fast similarity search
- Returns the most relevant chunk IDs (`faiss_id`)

---

### 3. Generation Layer (LLM)

- Uses retrieved chunks as context
- Generates answers grounded in experiment data
- Runs locally using Ollama (e.g., `llama3.2:3b`)

---

## Pipeline

### Step 1: Initialize Database

```bash
python -m src.data.init_db
```

Creates SQL tables.

---

### Step 2: Generate Data

```bash
python -m src.eval.generate_synthetic_data
```

- Inserts experiments into `experiments`
- Generates reports and stores them in `experiment_artifacts`

---

### Step 3: Build FAISS Index

```bash
python -m src.rag.build_faiss --rebuild
```

- Reads artifact content
- Splits text into chunks
- Embeds each chunk
- Stores vectors in FAISS
- Stores chunk metadata in `text_chunks`

---

### Step 4: Ask Questions

```bash
python -m src.rag.rag_answer "What is the accuracy of exp_001?"
```

Pipeline:

1. Query is embedded
2. FAISS retrieves top-k similar chunks
3. SQL fetches corresponding chunk text
4. Context is built with experiment references
5. LLM generates an answer using the context

---

## Chunking Strategy

- Uses a **character-based chunking approach**
- Default:
  - chunk size: 800–900 characters
  - overlap: ~120–150 characters

Purpose:
- Preserve context across boundaries
- Improve retrieval quality

---

## Retrieval Process

- Query → embedding
- FAISS → nearest vectors
- Vector IDs → mapped to `text_chunks`
- SQL → retrieves actual text and metadata

---

## Prompt Design

The system enforces:

- Use only provided context
- Include explicit metric values
- Allow small inference from context
- Avoid hallucination
- Provide suggestions if data is missing

---

## Current Limitations

- Character-based chunking (not token-aware)
- No metadata filtering before retrieval
- No reranking of retrieved results
- Fixed top-k retrieval
- N+1 SQL queries for experiment lookup
- Limited dataset (synthetic examples)

---

## Future Improvements

### Retrieval Improvements
- Token-based or semantic chunking
- Metadata filtering (by experiment, dataset, model)
- Hybrid search (SQL + vector)
- Reranking with cross-encoders or LLM

### Performance
- Batch SQL queries instead of per-chunk lookup
- Caching embeddings and results
- Support for larger FAISS indexes

### Data & Evaluation
- Add real experiment data
- Implement evaluation metrics (recall@k, precision@k)
- Add automated RAG evaluation pipeline

### System Design
- Expose as API (FastAPI service)
- Add authentication and multi-user support
- Logging and monitoring

### LLM Improvements
- Better prompt engineering
- Context compression
- Multi-step reasoning / agentic workflows

---

## Summary

This project demonstrates a complete end-to-end RAG pipeline:

- Structured + unstructured data storage (SQL)
- Efficient retrieval (FAISS)
- Grounded answer generation (LLM)

It serves as a strong foundation for building more advanced AI systems for experiment tracking and analysis.
