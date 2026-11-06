"""Turns text into embedding vectors with the configured sentence-transformers model."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config.settings import EMBED_QUERY_PREFIX, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once and reuse it.

    Returns:
        The EMBEDDING_MODEL sentence-transformers model.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for storage.

    Vectors are normalised to length 1 so cosine distance behaves well.

    Args:
        texts: Heading-prefixed chunk texts.

    Returns:
        One 384-dimensional vector per input text.
    """
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a search query.

    bge models are trained with an instruction in front of queries (but not
    documents); leaving it out noticeably hurts retrieval.

    Args:
        query: The user's question.

    Returns:
        One 384-dimensional normalised vector.
    """
    return _model().encode(EMBED_QUERY_PREFIX + query, normalize_embeddings=True).tolist()
