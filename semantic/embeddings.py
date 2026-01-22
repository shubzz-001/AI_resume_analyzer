from sentence_transformers import SentenceTransformer
import numpy as np
from semantic.cache import get_cached_embeddings, set_cached_embeddings

_model = None

def get_model() :
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(texts) :
    model = get_model()
    embedddings = []

    for text in texts :
        cached = get_cached_embeddings(text)
        if cached is not None:
            embedddings.append(cached)
        else :
            vec = model.encode(text, normalize_embeddings=True)
            set_cached_embeddings(text, vec)
            embedddings.append(vec)

    return np.array(embedddings)

def cosine_sim(a, b) :
    return float(np.dot(a, b))