"""Turns a document into chunks: split on H2, re-split long sections on H3, then fit to the model."""

import logging

from chunking.chunk_merger import merge_small_chunks
from chunking.context_header import build_heading_path
from chunking.header_splitter import split_on_heading
from chunking.markdown_syntax import has_content
from chunking.models import Chunk, Section, SourceDocument
from chunking.paragraph_splitter import pack_paragraphs
from chunking.text_size import count_tokens, count_words
from config.settings import (
    H3_SPLIT_THRESHOLD,
    MAX_EMBED_TOKENS,
    SECTION_HEADING_LEVEL,
    SUBSECTION_HEADING_LEVEL,
)

logger = logging.getLogger(__name__)


def _split_long_section(section: Section, split_on_h3: bool) -> list[Section]:
    """Apply the H3 rule: sections over H3_SPLIT_THRESHOLD words split on `###`.

    Text between the `##` heading and the first `###` becomes its own piece
    with subsection "", so the section's intro is never lost.

    Args:
        section: One `##` section.
        split_on_h3: False turns the H3 rule off (H2-only chunking).

    Returns:
        The H3 subsections, or the whole section as a single piece.
    """
    if split_on_h3 and count_words(section.text) > H3_SPLIT_THRESHOLD:
        return split_on_heading(section.text, SUBSECTION_HEADING_LEVEL)
    return [Section(heading="", text=section.text)]


def _fit_to_model(text: str, budget: int) -> list[str]:
    """Split text at paragraph (then line) boundaries if it exceeds the budget.

    Catches long sections with no H3s and H3 subsections that are still too big.

    Args:
        text: Raw Markdown of one section or subsection.
        budget: Tokens left for the chunk after its heading path.

    Returns:
        One or more parts, each within the budget unless a single line is longer.
    """
    if count_tokens(text) <= budget:
        return [text]
    return pack_paragraphs(text, budget)


def chunk_document(document: SourceDocument, split_on_h3: bool = True, merge_small: bool = True) -> list[Chunk]:
    """Chunk one document using structure-based rules.

    Steps: split on `##` → split sections over 400 words on `###` →
    split anything that, with its heading path, would exceed the model's
    token limit at paragraph boundaries → merge chunks under
    MIN_CHUNK_TOKENS into a same-section neighbour.

    Args:
        document: The loaded Markdown file.
        split_on_h3: False skips the H3 rule; the evaluation uses this to
            measure what the rule adds.
        merge_small: False keeps small chunks separate; the evaluation uses
            this to measure what merging adds.

    Returns:
        Chunks in document order with chunk_index 0, 1, 2, ...
    """
    chunks: list[Chunk] = []
    for section in split_on_heading(document.text, SECTION_HEADING_LEVEL):
        section_title = section.heading or document.doc_title
        for piece in _split_long_section(section, split_on_h3):
            heading_path = build_heading_path(document.doc_title, section_title, piece.heading)
            budget = MAX_EMBED_TOKENS - count_tokens(heading_path)
            for part in _fit_to_model(piece.text, budget):
                if not has_content(part):
                    continue
                token_count = count_tokens(part)
                if token_count > budget:
                    logger.warning(
                        "%s: chunk has %d tokens (budget %d) and will be truncated when embedded",
                        heading_path, token_count, budget,
                    )
                chunks.append(
                    Chunk(
                        text=part,
                        doc_title=document.doc_title,
                        section_title=section_title,
                        subsection=piece.heading,
                        chunk_index=len(chunks),
                        token_count=token_count,
                        source_path=document.source_path,
                    )
                )
    return merge_small_chunks(chunks) if merge_small else chunks
