import faiss
import numpy as np


def build_faiss_index(vectors):
    vectors = vectors.astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors)

    index.add(vectors)

    return index


def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)


def load_faiss_index(file_path):
    return faiss.read_index(file_path)


def search_faiss(index, query_vector, top_k=3):
    query_vector = query_vector.astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indexes = index.search(query_vector, top_k)

    return scores[0], indexes[0]