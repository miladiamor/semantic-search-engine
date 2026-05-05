# Semantic Search Engine (PyTorch)

##Project Overview

This project is a simple semantic search engine built using a pretrained embedding model.

It converts text into vectors and finds the most similar documents using cosine similarity.

---

## How It Works

The system follows this pipeline:

text → embedding → similarity → ranking

1. Documents are converted into vector embeddings
2. A user query is also converted into a vector
3. Cosine similarity is used to compare the query with all documents
4. Results are sorted and the most relevant ones are returned

---

## Technologies Used

- Python
- PyTorch (via Sentence Transformers)
- sentence-transformers
- NumPy
- scikit-learn

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt