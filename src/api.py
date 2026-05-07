import os
import numpy as np
from flask import Flask, request, jsonify, render_template

from embedder import Embedder
from search import search
from config import DOCUMENTS_PATH, EMBEDDINGS_PATH, FAISS_INDEX_PATH, TOP_K
from faiss_index import build_faiss_index, save_faiss_index, load_faiss_index


app = Flask(__name__, template_folder="../templates")


def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().splitlines()


embedder = Embedder()
documents = load_documents(DOCUMENTS_PATH)

if os.path.exists(EMBEDDINGS_PATH):
    print("Loading saved embeddings...")
    document_vectors = np.load(EMBEDDINGS_PATH)
else:
    print("Creating embeddings...")
    document_vectors = embedder.encode(documents)
    np.save(EMBEDDINGS_PATH, document_vectors)

if os.path.exists(FAISS_INDEX_PATH):
    print("Loading saved FAISS index...")
    faiss_index = load_faiss_index(FAISS_INDEX_PATH)
else:
    print("Creating FAISS index...")
    faiss_index = build_faiss_index(document_vectors)
    save_faiss_index(faiss_index, FAISS_INDEX_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json()

    query = data.get("query", "")

    if query.strip() == "":
        return jsonify({"error": "Query cannot be empty"}), 400

    results = search(
        query=query,
        documents=documents,
        faiss_index=faiss_index,
        embedder=embedder,
        top_k=TOP_K
    )

    response = [
        {"document": document, "score": float(score)}
        for document, score in results
    ]

    return jsonify({"query": query, "results": response})


if __name__ == "__main__":
    app.run(debug=True)