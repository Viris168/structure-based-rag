"""Runs vector and BM25 search, fuses them, and keeps the top-k chunks."""

from chromadb.api.models.Collection import Collection

from config.settings import SEARCH_CANDIDATES, TOP_K
from retrieval.bm25_search import bm25_search
from retrieval.models import RetrievedChunk
from retrieval.rrf import fuse_rankings
from retrieval.vector_search import vector_search
from vectorstore.chroma_client import get_collection


def hybrid_search_in(
    collection: Collection, query: str, include_heading_path: bool = True
) -> list[RetrievedChunk]:
    """Find the TOP_K chunks in a given collection most relevant to the query.

    Vector search catches paraphrases; BM25 catches exact names and terms.
    Each contributes SEARCH_CANDIDATES results, which are merged with RRF.

    Args:
        collection: The chunk collection to search.
        query: The user's question.
        include_heading_path: Passed to BM25; must match how the collection
            was embedded.

    Returns:
        Up to TOP_K chunks, best first, scored by RRF.
    """
    fused = fuse_rankings(
        [
            vector_search(collection, query, SEARCH_CANDIDATES),
            bm25_search(collection, query, SEARCH_CANDIDATES, include_heading_path),
        ]
    )
    return fused[:TOP_K]


def hybrid_search(query: str) -> list[RetrievedChunk]:
    """Find the TOP_K chunks in the main collection most relevant to the query.

    Args:
        query: The user's question.

    Returns:
        Up to TOP_K chunks, best first, scored by RRF.
    """
    return hybrid_search_in(get_collection(), query)
