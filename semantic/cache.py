import hashlib

_EMBED_CACHE = {}

def _hash_text(text: str) -> str :
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_embeddings(text) :
    key = _hash_text(text)
    return _EMBED_CACHE.get(key)

def set_cached_embeddings(text, vector) :
    key = _hash_text(text)
    _EMBED_CACHE[key] = vector

def cache_size() :
    return len(_EMBED_CACHE)