from sentence_transformers import SentenceTransformer
from config import MODEL_NAME
class Embedder:
    def __init__(self):
        print("Loading model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Model loaded!")

    def encode(self, texts):
        return self.model.encode(texts)