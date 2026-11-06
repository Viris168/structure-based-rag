"""Builds the heading path that is prepended to a chunk before embedding."""

from chunking.metadata import Metadata
from chunking.models import Chunk
from config.settings import HEADING_PATH_SEPARATOR


def build_heading_path(doc_title: str, section_title: str, subsection: str) -> str:
    """Join doc title, section, and subsection into one breadcrumb.

    Repeated or empty parts are skipped (preamble chunks use the doc title as
    their section, so it would otherwise appear twice).

    Args:
        doc_title: The document's title.
        section_title: The `##` heading, or the doc title for the preamble.
        subsection: The `###` heading, or "".

    Returns:
        e.g. "MCP Documentation > Installation > Docker".
    """
    parts: list[str] = []
    for part in (doc_title, section_title, subsection):
        if part and part not in parts:
            parts.append(part)
    return HEADING_PATH_SEPARATOR.join(parts)


def build_embedding_text(chunk: Chunk, include_heading_path: bool = True) -> str:
    """Return the text the model embeds: heading path, blank line, raw chunk.

    The stored document stays the raw chunk; only the vector sees the prefix.

    Args:
        chunk: The chunk to embed.
        include_heading_path: False embeds the raw chunk alone; the evaluation
            uses this to measure what the heading path adds.

    Returns:
        Heading path followed by the chunk text.
    """
    if not include_heading_path:
        return chunk.text
    heading_path = build_heading_path(chunk.doc_title, chunk.section_title, chunk.subsection)
    return f"{heading_path}\n\n{chunk.text}"


def build_stored_chunk_text(text: str, metadata: Metadata, include_heading_path: bool = True) -> str:
    """Rebuild a stored chunk's embedded form from its raw text and metadata.

    Search results carry the raw text and metadata, not a Chunk; BM25 and the
    reranker use this so they see the same text the embedding model saw.

    Args:
        text: Raw chunk text as stored in ChromaDB.
        metadata: The chunk's stored metadata.
        include_heading_path: False returns the raw text alone.

    Returns:
        Heading path, blank line, chunk text (or just the chunk text).
    """
    if not include_heading_path:
        return text
    heading_path = build_heading_path(
        str(metadata["doc_title"]), str(metadata["section_title"]), str(metadata["subsection"])
    )
    return f"{heading_path}\n\n{text}"
