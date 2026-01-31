import hashlib
from typing import Optional, Dict
import logging
import numpy as np
from collections import OrderedDict

from config import EMBEDDING_CACHE_SIZE, ENABLE_CACHE

logger = logging.getLogger(__name__)

# Global cache storage (LRU cache)
_embedding_cache: OrderedDict = OrderedDict()
_cache_stats = {
    'hits': 0,
    'misses': 0,
    'size': 0
}


def _hash_text(text: str) -> str:
    """
    Generate hash for text to use as cache key.
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_cached_embedding(text: str) -> Optional[np.ndarray]:
    """
    Retrieve embedding from cache.
    """
    if not ENABLE_CACHE:
        return None

    key = _hash_text(text)

    if key in _embedding_cache:
        # Move to end (most recently used)
        _embedding_cache.move_to_end(key)
        _cache_stats['hits'] += 1
        logger.debug(f"Cache hit for text hash: {key[:8]}...")
        return _embedding_cache[key]

    _cache_stats['misses'] += 1
    return None


def set_cached_embedding(text: str, embedding: np.ndarray) -> None:
    """
    Store embedding in cache.
    """
    if not ENABLE_CACHE:
        return

    key = _hash_text(text)

    # Remove oldest if cache is full
    if len(_embedding_cache) >= EMBEDDING_CACHE_SIZE:
        # Remove least recently used (first item)
        oldest_key = next(iter(_embedding_cache))
        _embedding_cache.pop(oldest_key)
        logger.debug(f"Cache full, removed oldest entry")

    # Add to cache
    _embedding_cache[key] = embedding
    _cache_stats['size'] = len(_embedding_cache)
    logger.debug(f"Cached embedding for text hash: {key[:8]}...")


def clear_cache() -> None:
    """Clear all cached embeddings."""
    global _embedding_cache, _cache_stats

    _embedding_cache.clear()
    _cache_stats = {
        'hits': 0,
        'misses': 0,
        'size': 0
    }

    logger.info("Cache cleared")


def get_cache_size() -> int:
    """
    Get current cache size.
    """
    return len(_embedding_cache)


def get_cache_stats() -> Dict[str, any]:
    """
    Get cache statistics.
    """
    total_requests = _cache_stats['hits'] + _cache_stats['misses']
    hit_rate = (_cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0

    return {
        'size': len(_embedding_cache),
        'max_size': EMBEDDING_CACHE_SIZE,
        'hits': _cache_stats['hits'],
        'misses': _cache_stats['misses'],
        'hit_rate': round(hit_rate, 2),
        'total_requests': total_requests,
        'enabled': ENABLE_CACHE
    }


def cache_info() -> str:
    """
    Get formatted cache information.
    """
    stats = get_cache_stats()

    info = f"""
Cache Information:
  Status: {'Enabled' if stats['enabled'] else 'Disabled'}
  Size: {stats['size']} / {stats['max_size']}
  Hits: {stats['hits']}
  Misses: {stats['misses']}
  Hit Rate: {stats['hit_rate']}%
  Total Requests: {stats['total_requests']}
    """

    return info.strip()


def optimize_cache(max_size: Optional[int] = None) -> None:
    """
    Optimize cache by removing least used entries.
    """
    if max_size is not None and max_size < len(_embedding_cache):
        # Keep only the most recent entries
        while len(_embedding_cache) > max_size:
            oldest = next(iter(_embedding_cache))
            _embedding_cache.pop(oldest)

        logger.info(f"Cache optimized to {max_size} entries")


def cache_size() -> int:
    """
    Backward compatible function for cache size.
    """
    return get_cache_size()


class EmbeddingCache:
    """
    Class-based cache manager for advanced usage.
    """

    def __init__(self, max_size: int = EMBEDDING_CACHE_SIZE):
        """
        Initialize cache manager.
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0
        }

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = _hash_text(text)

        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats['hits'] += 1
            return self.cache[key]

        self.stats['misses'] += 1
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Set embedding in cache."""
        key = _hash_text(text)

        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = embedding

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.stats = {'hits': 0, 'misses': 0}

    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)

    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.stats['hits'] + self.stats['misses']
        return (self.stats['hits'] / total * 100) if total > 0 else 0


# Module-level cache instance
default_cache = EmbeddingCache()