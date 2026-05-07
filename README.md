# Semantic Search Engine (PyTorch + FAISS + Flask)

## Project Overview

This project is a semantic search engine built using transformer embeddings.

It converts text into vector embeddings and retrieves the most semantically similar documents using FAISS vector search.

The project also includes:

- FAISS vector indexing
- Flask REST API
- Interactive web interface
- Persistent embedding storage

---

## Features

- Semantic search using embeddings
- Transformer-based text vectorization
- Cosine similarity retrieval
- FAISS vector indexing for scalable search
- Flask backend API
- Web frontend interface
- Saved embeddings and FAISS indexes
- Modular project structure

---

## Technologies Used

- Python
- PyTorch
- Sentence Transformers
- FAISS
- Flask
- NumPy
- scikit-learn

---

## Project Structure

```text
semantic_search_engine/
│
├── data/
├── embeddings/
├── src/
├── templates/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/miladiamor/semantic-search-engine.git
cd semantic-search-engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Terminal Search

```bash
python src/main.py
```

---

## Run Flask API + Web App

```bash
python src/api.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

## Example API Request

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/search" `
-Method POST `
-ContentType "application/json" `
-Body '{"query":"fix my bicycle"}'
```

---

## How It Works

1. Documents are converted into embeddings using a transformer model.
2. Embeddings are normalized and indexed using FAISS.
3. User queries are converted into embeddings.
4. FAISS retrieves the nearest semantic matches.
5. Results are returned through the API or web interface.

---

## Future Improvements

- Larger document datasets
- PDF document ingestion
- Docker deployment
- Cloud deployment
- Hybrid keyword + semantic search
- Multi-language support

---

## Author

Milad Amor