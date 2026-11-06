"""Re-scores search candidates by reading the query and each chunk together."""

from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from chunking.context_header import build_stored_chunk_text
from config.settings import RERANKER_MAX_TOKENS, RERANKER_MODEL
from retrieval.models import RetrievedChunk


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    """Load the reranker once and reuse it.

    Returns:
        The RERANKER_MODEL cross-encoder.
    """
    return CrossEncoder(RERANKER_MODEL, max_length=RERANKER_MAX_TOKENS)


def rerank(
    query: str, chunks: list[RetrievedChunk], top_k: int, include_heading_path: bool = True
) -> list[RetrievedChunk]:
    """Re-order chunks by how well each one answers the query.

    Vector search compares two separately made vectors, so it can only judge
    "same topic". A cross-encoder reads the query and the chunk side by side,
    so it can tell that "Usage > Set default node version" answers "which node
    does every new shell use?" better than the general "Usage" intro does.
    It is too slow to run on every chunk, so it only re-scores the candidates
    that hybrid search already found.

    Args:
        query: The user's question.
        chunks: Candidates from hybrid search.
        top_k: How many to keep.
        include_heading_path: Whether the reranker sees each chunk's heading path.

    Returns:
        Up to top_k chunks, best first. Score is the reranker's relevance score.
    """
    if not chunks:
        return []
    pairs = [(query, build_stored_chunk_text(c.text, c.metadata, include_heading_path)) for c in chunks]
    scores = _model().predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [replace(chunk, score=float(score)) for score, chunk in ranked[:top_k]]
