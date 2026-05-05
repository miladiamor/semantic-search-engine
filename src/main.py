import os
import sys
import numpy as np

from embedder import Embedder
from search import search
from config import DOCUMENTS_PATH, EMBEDDINGS_PATH, TOP_K



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
            document_vectors=document_vectors,
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