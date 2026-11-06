"""Formats source citations and picks out the ones an answer actually used."""

from chunking.context_header import build_heading_path
from chunking.metadata import Metadata
from retrieval.models import RetrievedChunk


def format_citation(metadata: Metadata) -> str:
    """Build the citation label for one chunk.

    Args:
        metadata: The chunk's stored metadata.

    Returns:
        e.g. "[Source: MCP Documentation · Section: Installation > Docker]".
    """
    section = build_heading_path("", str(metadata["section_title"]), str(metadata["subsection"]))
    return f"[Source: {metadata['doc_title']} · Section: {section}]"


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
