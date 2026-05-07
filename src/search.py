from faiss_index import search_faiss


def search(query, documents, faiss_index, embedder, top_k=3):
    query_vector = embedder.encode([query])

    scores, indexes = search_faiss(
        index=faiss_index,
        query_vector=query_vector,
        top_k=top_k
    )

    results = []

    for score, doc_index in zip(scores, indexes):
        document = documents[doc_index]
        results.append((document, score))

    return results