"""Folds chunks too small to stand alone into a neighbour from the same `##` section."""

from dataclasses import replace

from chunking.context_header import build_heading_path
from chunking.models import Chunk
from chunking.text_size import count_tokens
from config.settings import MAX_EMBED_TOKENS, MIN_CHUNK_TOKENS, SUBSECTION_JOINER


def _join_subsections(first: str, second: str) -> str:
    """Combine two subsection labels, skipping empty and repeated ones.

    Args:
        first: Subsection of the earlier chunk ("" for a section intro).
        second: Subsection of the later chunk.

    Returns:
        e.g. "Listing Versions / Restoring PATH", or the single non-empty label.
    """
    parts: list[str] = []
    for label in (*first.split(SUBSECTION_JOINER), *second.split(SUBSECTION_JOINER)):
        if label and label not in parts:
            parts.append(label)
    return SUBSECTION_JOINER.join(parts)


def _try_combine(first: Chunk, second: Chunk) -> Chunk | None:
    """Join two consecutive chunks if they share a section and still fit the model.

    Chunks from different `##` sections are never joined: the result would
    carry one section's citation label for another section's text.

    Args:
        first: The earlier chunk.
        second: The chunk right after it.

    Returns:
        The combined chunk, or None if they cannot be joined.
    """
    if first.source_path != second.source_path or first.section_title != second.section_title:
        return None
    text = f"{first.text}\n\n{second.text}"
    subsection = _join_subsections(first.subsection, second.subsection)
    heading_path = build_heading_path(first.doc_title, first.section_title, subsection)
    token_count = count_tokens(text)
    if token_count > MAX_EMBED_TOKENS - count_tokens(heading_path):
        return None
    return replace(first, text=text, subsection=subsection, token_count=token_count)


def merge_small_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Merge every chunk under MIN_CHUNK_TOKENS into a same-section neighbour.

    A small chunk joins the chunk before it if possible, otherwise the chunk
    after it (e.g. a short section intro joins its first subsection). A small
    chunk with no same-section neighbour, such as a one-line `## License`,
    is kept as it is.

    Args:
        chunks: One document's chunks, in order.

    Returns:
        The chunks after merging, re-numbered 0, 1, 2, ...
    """
    merged: list[Chunk] = []
    waiting: Chunk | None = None  # a small chunk that could not join the one before it
    for chunk in chunks:
        if waiting is not None:
            combined = _try_combine(waiting, chunk)
            if combined is not None:
                chunk = combined
            else:
                merged.append(waiting)
            waiting = None
        if chunk.token_count < MIN_CHUNK_TOKENS:
            combined = _try_combine(merged[-1], chunk) if merged else None
            if combined is not None:
                merged[-1] = combined
            else:
                waiting = chunk
            continue
        merged.append(chunk)
    if waiting is not None:
        merged.append(waiting)
    return [replace(chunk, chunk_index=index) for index, chunk in enumerate(merged)]
