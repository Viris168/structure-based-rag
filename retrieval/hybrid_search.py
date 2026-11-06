"""Runs vector and BM25 search, fuses them, and keeps the top-k chunks."""

from chromadb.api.models.Collection import Collection

from config.settings import RERANK_CANDIDATES, SEARCH_CANDIDATES, TOP_K
from retrieval.bm25_search import bm25_search
from retrieval.models import RetrievedChunk
from retrieval.reranker import rerank
from retrieval.rrf import fuse_rankings
from retrieval.vector_search import vector_search
from vectorstore.chroma_client import get_collection


def hybrid_search_in(
    collection: Collection, query: str, include_heading_path: bool = True, use_reranker: bool = True
) -> list[RetrievedChunk]:
    """Find the TOP_K chunks in a given collection most relevant to the query.

    Vector search catches paraphrases; BM25 catches exact names and terms.
    Each contributes SEARCH_CANDIDATES results, which are merged with RRF.
    The reranker then re-scores the best RERANK_CANDIDATES and keeps TOP_K.

    Args:
        collection: The chunk collection to search.
        query: The user's question.
        include_heading_path: Passed to BM25 and the reranker; must match how
            the collection was embedded.
        use_reranker: False returns the RRF order directly; the evaluation
            uses this to measure what the reranker adds.

    Returns:
        Up to TOP_K chunks, best first, scored by the reranker (or by RRF).
    """
    fused = fuse_rankings(
        [
            vector_search(collection, query, SEARCH_CANDIDATES),
            bm25_search(collection, query, SEARCH_CANDIDATES, include_heading_path),
        ]
    )
    if not use_reranker:
        return fused[:TOP_K]
    return rerank(query, fused[:RERANK_CANDIDATES], TOP_K, include_heading_path)


def hybrid_search(query: str) -> list[RetrievedChunk]:
    """Find the TOP_K chunks in the main collection most relevant to the query.

    Args:
        query: The user's question.

    Returns:
        Up to TOP_K chunks, best first, scored by the reranker.
    """
    return hybrid_search_in(get_collection(), query)
