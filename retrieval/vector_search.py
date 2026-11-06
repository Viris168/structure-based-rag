"""Finds the chunks whose embeddings are closest in meaning to the query."""

from chromadb.api.models.Collection import Collection

from retrieval.models import RetrievedChunk
from vectorstore.embedder import embed_query


def vector_search(collection: Collection, query: str, n_results: int) -> list[RetrievedChunk]:
    """Rank chunks by cosine similarity to the query.

    Args:
        collection: The chunk collection.
        query: The user's question.
        n_results: Maximum number of chunks to return.

    Returns:
        Up to n_results chunks, most similar first. Score is cosine
        similarity (1 - cosine distance).
    """
    n_results = min(n_results, collection.count())
    if n_results == 0:
        return []

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, score=1.0 - distance)
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
