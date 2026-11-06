"""Merges several ranked result lists with Reciprocal Rank Fusion."""

from config.settings import RRF_K
from retrieval.models import RetrievedChunk


def fuse_rankings(rankings: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Combine ranked lists into one, using only each chunk's rank in each list.

    Each list adds 1 / (RRF_K + rank) to a chunk's score (rank starts at 1).
    Ranks are used instead of raw scores because cosine and BM25 scores are on
    different scales and cannot be added directly.

    Args:
        rankings: Result lists, each ordered best first.

    Returns:
        Every chunk that appears in any list, once, ordered by fused score.
        Score is the RRF score.
    """
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunks_by_id.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=chunks_by_id[chunk_id].text,
            metadata=chunks_by_id[chunk_id].metadata,
            score=fused_scores[chunk_id],
        )
        for chunk_id in ordered_ids
    ]
