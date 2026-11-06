"""Builds the metadata dict stored next to each chunk in ChromaDB."""

from chunking.models import Chunk
from chunking.text_size import count_words
from vectorstore.doc_hasher import hash_text

Metadata = dict[str, str | int]


def build_metadata(chunk: Chunk, file_hash: str) -> Metadata:
    """Create the metadata for one chunk.

    ChromaDB rejects None values, so a missing subsection is stored as "".

    Args:
        chunk: The chunk being stored.
        file_hash: Hash of the whole source file, used to detect edits on re-ingest.

    Returns:
        Fields for filtering, citations, and change detection.
    """
    return {
        "doc_title": chunk.doc_title,
        "section_title": chunk.section_title,
        "subsection": chunk.subsection,
        "chunk_index": chunk.chunk_index,
        "word_count": count_words(chunk.text),
        "token_count": chunk.token_count,
        "content_hash": hash_text(chunk.text),
        "source_path": chunk.source_path,
        "file_hash": file_hash,
    }
