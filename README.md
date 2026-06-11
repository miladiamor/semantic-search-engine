# Semantic Search Engine

A prototype semantic document retrieval system built with Python and PyTorch. Unlike keyword search, this system understands the *meaning* of a query and returns semantically relevant results even when exact words don't match.

## What it does

- Encodes documents and queries into dense vector representations using transformer-based sentence embeddings
- Uses **FAISS** for fast approximate nearest-neighbor search — no brute-force comparison needed even as the corpus grows
- Ranks results by cosine similarity between query and document embeddings
- Flask interface for testing and demonstrating retrieval

## How it works

```
Query text
    ↓
Transformer model (sentence embeddings)
    ↓
Dense vector representation
    ↓
FAISS index search (approximate nearest neighbors)
    ↓
Cosine similarity ranking
    ↓
Top-k semantically relevant results
```

## Why this matters

Traditional keyword search fails when the user says "cheap flights" but the document says "low-cost travel options." Semantic search captures the meaning, not just the words. This is the core concept behind modern RAG (Retrieval-Augmented Generation) pipelines used in production AI systems.

## Tech stack

- **PyTorch** — transformer model inference
- **Sentence Transformers** — pre-trained embedding models
- **FAISS** — efficient similarity search index
- **Flask** — lightweight web interface
- **Python** — core implementation

## How to run

```bash
# 1. Install dependencies
pip install torch sentence-transformers faiss-cpu flask

# 2. Run the app
python app.py

# 3. Open in browser
http://localhost:5000
```

## Project context

Built as part of coursework at Berliner Hochschule für Technik (BHT), 2026.

**Author:** Amor Miladi
