"""Formats source citations and picks out the ones an answer actually used."""

import re

from chunking.context_header import build_heading_path
from chunking.metadata import Metadata
from retrieval.models import RetrievedChunk

# Code blocks and inline code, where "[1]" means array indexing, not a citation.
_CODE_PATTERN = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
# One or more adjacent citation markers: "[1]", "[1][3]", "[1, 3]", "[2] [4]".
# Not preceded by a letter, digit, "_" or ")", so "items[1]" or "f(x)[0]" is left alone.
_MARKER_GROUP_PATTERN = re.compile(r"(?<![\w)])\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*")
_NUMBER_PATTERN = re.compile(r"\d+")


def format_citation(metadata: Metadata) -> str:
    """Build the citation label for one chunk.

    Args:
        metadata: The chunk's stored metadata.

    Returns:
        e.g. "[Source: MCP Documentation · Section: Installation > Docker]".
    """
    section = build_heading_path("", str(metadata["section_title"]), str(metadata["subsection"]))
    return f"[Source: {metadata['doc_title']} · Section: {section}]"


def _labels_for_markers(markers: str, chunks: list[RetrievedChunk]) -> str:
    """Turn one group of citation markers into the matching citation labels.

    Numbers outside 1..len(chunks) point at no excerpt, so they are dropped
    rather than shown as a citation to a source that does not exist.

    Args:
        markers: e.g. "[1][3]" or "[1, 3]".
        chunks: The excerpts in the order they were numbered in the prompt.

    Returns:
        e.g. "[Source: Doc · Section: A] [Source: Doc · Section: B]"; repeats removed.
    """
    labels: list[str] = []
    for number in (int(n) for n in _NUMBER_PATTERN.findall(markers)):
        if 1 <= number <= len(chunks):
            label = format_citation(chunks[number - 1].metadata)
            if label not in labels:
                labels.append(label)
    return " ".join(labels)


def _expand_in_prose(prose: str, chunks: list[RetrievedChunk]) -> str:
    """Replace every citation marker group in a stretch of text that contains no code.

    Args:
        prose: Part of the answer outside code blocks and inline code.
        chunks: The excerpts in the order they were numbered in the prompt.

    Returns:
        The text with markers replaced by citation labels.
    """
    return _MARKER_GROUP_PATTERN.sub(lambda match: _labels_for_markers(match.group(), chunks), prose)


def expand_citation_numbers(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Replace the model's numbered citations ("[2]") with full labels.

    The model only has to copy a number, which it cannot misspell; the exact
    label is filled in here, so every citation names a real retrieved chunk.
    Code is left untouched, so `items[1]` in a code sample stays as it is.

    Args:
        answer: The model's reply, citing excerpts as [1], [2], ...
        chunks: The excerpts in the order they were numbered in the prompt.

    Returns:
        The reply with each marker group replaced by citation labels.
    """
    parts: list[str] = []
    position = 0
    for code in _CODE_PATTERN.finditer(answer):
        parts.append(_expand_in_prose(answer[position : code.start()], chunks))
        parts.append(code.group())
        position = code.end()
    parts.append(_expand_in_prose(answer[position:], chunks))
    return "".join(parts)


def cited_sources(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """List the citations from the retrieved chunks that appear in the answer.

    Args:
        answer: The model's reply.
        chunks: The chunks that were given to the model.

    Returns:
        Distinct citation labels found in the answer, in retrieval order.
    """
    citations: list[str] = []
    for chunk in chunks:
        citation = format_citation(chunk.metadata)
        if citation in answer and citation not in citations:
            citations.append(citation)
    return citations
