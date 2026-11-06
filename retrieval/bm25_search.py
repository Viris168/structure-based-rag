"""Finds chunks by keyword overlap with the query (BM25)."""

import re

from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from chunking.context_header import build_stored_chunk_text
from chunking.metadata import Metadata
from retrieval.models import RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

# Built indexes, keyed by collection name and heading-path setting. Each entry
# remembers the chunk IDs it was built from; IDs contain a content hash, so
# any add, edit, or removal makes the entry stale and it is rebuilt.
_index_cache: dict[tuple[str, bool], tuple[list[str], BM25Okapi]] = {}


def _tokenize(text: str) -> list[str]:
    """Lower-case the text and split it into word tokens.

    Identifiers like `get_or_create_collection` are kept whole (so an exact
    name matches strongly) and also split on "_" (so "collection" matches too).

    Args:
        text: Any text.

    Returns:
        The tokens, in order.
    """
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text.lower()):
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return tokens


def _get_index(
    collection_name: str,
    ids: list[str],
    texts: list[str],
    metadatas: list[Metadata],
    include_heading_path: bool,
) -> BM25Okapi:
    """Return a BM25 index over the given chunks, rebuilding only if they changed.

    Args:
        collection_name: Name of the collection the chunks came from.
        ids: Chunk IDs, in collection order.
        texts: Raw chunk texts, same order.
        metadatas: Chunk metadata, same order.
        include_heading_path: Whether heading paths are indexed with the text.

    Returns:
        The BM25 index; position i corresponds to ids[i].
    """
    key = (collection_name, include_heading_path)
    cached = _index_cache.get(key)
    if cached is None or cached[0] != ids:
        corpus = [_tokenize(build_stored_chunk_text(t, m, include_heading_path)) for t, m in zip(texts, metadatas)]
        cached = (ids, BM25Okapi(corpus))
        _index_cache[key] = cached
    return cached[1]


def bm25_search(
    collection: Collection, query: str, n_results: int, include_heading_path: bool = True
) -> list[RetrievedChunk]:
    """Rank chunks by BM25 keyword score against the query.

    Chunks that share no words with the query are left out, so they get no
    rank in the fusion step.

    Args:
        collection: The chunk collection.
        query: The user's question.
        n_results: Maximum number of chunks to return.
        include_heading_path: False matches against raw chunk text only; the
            evaluation uses this to measure what the heading path adds.

    Returns:
        Up to n_results chunks, best keyword match first. Score is the BM25 score.
    """
    stored = collection.get(include=["documents", "metadatas"])
    ids, texts, metadatas = stored["ids"], stored["documents"], stored["metadatas"]
    if not ids:
        return []

    index = _get_index(collection.name, ids, texts, metadatas, include_heading_path)
    scores = index.get_scores(_tokenize(query))
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    return [
        RetrievedChunk(chunk_id=ids[i], text=texts[i], metadata=metadatas[i], score=float(scores[i]))
        for i in ranked[:n_results]
        if scores[i] > 0
    ]
