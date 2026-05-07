import os
import sys
import numpy as np

from embedder import Embedder
from search import search
from config import DOCUMENTS_PATH, EMBEDDINGS_PATH, TOP_K, FAISS_INDEX_PATH
from faiss_index import build_faiss_index, save_faiss_index, load_faiss_index


def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().splitlines()


def main():
    embedder = Embedder()

    documents = load_documents(DOCUMENTS_PATH)
    refresh_embeddings = "--refresh" in sys.argv

    if refresh_embeddings or not os.path.exists(EMBEDDINGS_PATH):
        print("Creating embeddings...")
        document_vectors = embedder.encode(documents)
        np.save(EMBEDDINGS_PATH, document_vectors)
        print("Embeddings saved!")
    else:
        print("Loading saved embeddings...")
        document_vectors = np.load(EMBEDDINGS_PATH)
    
    if refresh_embeddings or not os.path.exists(FAISS_INDEX_PATH):
        print("Creating FAISS index...")
        faiss_index = build_faiss_index(document_vectors)
        save_faiss_index(faiss_index, FAISS_INDEX_PATH)
        print("FAISS index saved!")
    else:
        print("Loading saved FAISS index...")
        faiss_index = load_faiss_index(FAISS_INDEX_PATH)

    while True:
        query = input("\nEnter your search query or type 'exit' to quit: ")
        if query.strip() == "":
            print("Please enter a search query.")
            continue
        
        if query.lower() == "exit":
            print("Goodbye!")
            break

        results = search(
            query=query,
            documents=documents,
            faiss_index=faiss_index,
            embedder=embedder,
            top_k=TOP_K
        )

        print("\n" + "=" * 50)
        print(f"Search query: {query}")
        print("=" * 50)

        print(f"\nTop {TOP_K} results:\n")

        for index, (document, score) in enumerate(results, start=1):
            print(f"{index}. {document}")
            print(f"   Similarity score: {score:.4f}\n")


if __name__ == "__main__":
    main()