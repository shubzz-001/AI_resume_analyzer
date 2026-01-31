from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import logging
import hashlib
import pickle
from pathlib import Path

from config import EMBEDDING_MODEL, EMBEDDING_CACHE_SIZE, ENABLE_CACHE
from semantic.cache import get_cached_embedding, set_cached_embedding, get_cache_stats

logger = logging.getLogger(__name__)

# Global model cache (singleton pattern)
_model_instance = None


def get_model() -> SentenceTransformer:

    global _model_instance

    if _model_instance is None:
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _model_instance = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")

    return _model_instance


def embed_text(
        text: Union[str, List[str]],
        normalize: bool = True,
        use_cache: bool = True
) -> np.ndarray:
    """
    Generate embeddings for text(s).
    """
    # Handle single text vs list
    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    if not texts:
        return np.array([])

    # Get model
    model = get_model()

    embeddings = []
    texts_to_embed = []
    indices_to_embed = []

    # Check cache if enabled
    if use_cache and ENABLE_CACHE:
        for idx, txt in enumerate(texts):
            cached = get_cached_embedding(txt)
            if cached is not None:
                embeddings.append((idx, cached))
            else:
                texts_to_embed.append(txt)
                indices_to_embed.append(idx)
    else:
        texts_to_embed = texts
        indices_to_embed = list(range(len(texts)))

    # Generate embeddings for uncached texts
    if texts_to_embed:
        try:
            new_embeddings = model.encode(
                texts_to_embed,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )

            # Cache new embeddings
            if use_cache and ENABLE_CACHE:
                for txt, emb in zip(texts_to_embed, new_embeddings):
                    set_cached_embedding(txt, emb)

            # Add to results
            for idx, emb in zip(indices_to_embed, new_embeddings):
                embeddings.append((idx, emb))

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    # Sort by original index and extract embeddings
    embeddings.sort(key=lambda x: x[0])
    result = np.array([emb for _, emb in embeddings])

    # Return single embedding if single text
    if is_single:
        return result[0]

    return result


def embed(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Convenience function for embedding (backward compatible).
    """
    return embed_text(texts, normalize=True, use_cache=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    Assumes embeddings are already normalized.
    """
    if a.ndim == 1 and b.ndim == 1:
        # Simple dot product for normalized vectors
        return float(np.dot(a, b))
    else:
        # Handle batch
        return float(np.dot(a, b.T))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Alias for cosine_similarity (backward compatible).
    """
    return cosine_similarity(a, b)


def batch_cosine_similarity(
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity between one query and multiple corpus embeddings.
    """
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Calculate similarities
    similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()

    return similarities


def semantic_search(
        query: str,
        corpus: List[str],
        top_k: int = 5,
        threshold: float = 0.0
) -> List[dict]:
    """
    Perform semantic search on a corpus.
    """
    if not corpus:
        return []

    # Generate embeddings
    query_emb = embed_text(query)
    corpus_embs = embed_text(corpus)

    # Calculate similarities
    similarities = batch_cosine_similarity(query_emb, corpus_embs)

    # Get top K
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score >= threshold:
            results.append({
                'text': corpus[idx],
                'score': float(score),
                'index': int(idx)
            })

    return results


def get_text_similarity(text1: str, text2: str) -> float:
    """
    Get similarity between two texts.
    """
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)

    return cosine_similarity(emb1, emb2)


def get_embedding_dimension() -> int:
    """
    Get dimension of embeddings.
    """
    model = get_model()
    return model.get_sentence_embedding_dimension()


def clear_embedding_cache():
    """Clear the embedding cache."""
    from semantic.cache import clear_cache
    clear_cache()
    logger.info("Embedding cache cleared")


def save_embeddings(
        embeddings: np.ndarray,
        filepath: Union[str, Path]
):
    """
    Save embeddings to file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

    logger.info(f"Saved embeddings to {filepath}")


def load_embeddings(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load embeddings from file.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")

    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)

    logger.info(f"Loaded embeddings from {filepath}")
    return embeddings


class EmbeddingGenerator:
    """
    Class-based embedding generator for advanced usage.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(
            self,
            texts: Union[str, List[str]],
            normalize: bool = True
    ) -> np.ndarray:

        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

    def similarity(self, text1: str, text2: str) -> float:

        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1, emb2))

    def search(
            self,
            query: str,
            corpus: List[str],
            top_k: int = 5
    ) -> List[dict]:

        query_emb = self.encode(query)
        corpus_embs = self.encode(corpus)

        similarities = batch_cosine_similarity(query_emb, corpus_embs)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                'text': corpus[idx],
                'score': float(similarities[idx]),
                'index': int(idx)
            }
            for idx in top_indices
        ]


# Backward compatibility functions
def get_embeddings(texts: List[str]) -> np.ndarray:
    """Legacy function name."""
    return embed_text(texts)