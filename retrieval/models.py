"""Data container returned by every search function."""

from dataclasses import dataclass

from chunking.metadata import Metadata


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by a search, with the score that ranked it.

    Attributes:
        chunk_id: The chunk's ChromaDB ID.
        text: Raw Markdown of the chunk, exactly as stored.
        metadata: The chunk's stored metadata (titles, index, hashes).
        score: Ranking score; higher is better. Its meaning depends on the
            search that produced it (cosine similarity, BM25, or RRF).
    """

    chunk_id: str
    text: str
    metadata: Metadata
    score: float
