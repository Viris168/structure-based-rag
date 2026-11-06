"""Writes chunks to ChromaDB one document at a time, replacing stale versions."""

from chromadb.api.models.Collection import Collection

from chunking.metadata import Metadata
from config.settings import CHUNK_ID_HASH_LENGTH


def _chunk_id(metadata: Metadata) -> str:
    """Build a stable ID: same file, position, and content always give the same ID.

    Args:
        metadata: The chunk's metadata.

    Returns:
        e.g. "DOCUMENTATION.md::3::a1b2c3d4e5f6".
    """
    content_hash = str(metadata["content_hash"])[:CHUNK_ID_HASH_LENGTH]
    return f"{metadata['source_path']}::{metadata['chunk_index']}::{content_hash}"


def get_stored_file_hash(collection: Collection, source_path: str) -> str | None:
    """Look up the file hash recorded the last time this document was ingested.

    Args:
        collection: The chunk collection.
        source_path: Filename relative to the docs folder.

    Returns:
        The stored hash, or None if the document has never been ingested.
    """
    result = collection.get(where={"source_path": source_path}, limit=1, include=["metadatas"])
    metadatas = result["metadatas"]
    return str(metadatas[0]["file_hash"]) if metadatas else None


def replace_document_chunks(
    collection: Collection,
    source_path: str,
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[Metadata],
) -> None:
    """Delete every old chunk for a document, then add its new chunks.

    Replacing the whole document means an edited or removed section never
    leaves a stale chunk behind.

    Args:
        collection: The chunk collection.
        source_path: Filename relative to the docs folder.
        documents: Raw chunk texts (what search results will show).
        embeddings: One vector per chunk.
        metadatas: One metadata dict per chunk.
    """
    collection.delete(where={"source_path": source_path})
    if not documents:
        return
    collection.add(
        ids=[_chunk_id(metadata) for metadata in metadatas],
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def remove_documents_not_in(collection: Collection, present_paths: set[str]) -> list[str]:
    """Delete chunks belonging to files that no longer exist in the docs folder.

    Args:
        collection: The chunk collection.
        present_paths: source_path of every document found on disk this run.

    Returns:
        The source paths that were removed, sorted.
    """
    stored_paths = {str(m["source_path"]) for m in collection.get(include=["metadatas"])["metadatas"]}
    removed = sorted(stored_paths - present_paths)
    for source_path in removed:
        collection.delete(where={"source_path": source_path})
    return removed
