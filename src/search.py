from sklearn.metrics.pairwise import cosine_similarity


def search(query, documents, document_vectors, embedder, top_k=3):
    query_vector = embedder.encode([query])

    similarities = cosine_similarity(query_vector, document_vectors)[0]

    results = sorted(
        zip(documents, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    return results[:top_k]