"""Decides whether a retrieved chunk is the one an evaluation question expects."""

from eval.questions import EvalQuestion
from retrieval.models import RetrievedChunk


def is_expected_chunk(question: EvalQuestion, chunk: RetrievedChunk) -> bool:
    """Check whether a chunk contains the section the question expects.

    A chunk counts when it comes from the expected doc and `##` section and,
    if a `###` subsection is expected, it either is that subsection or still
    contains it (as happens when the section was not H3-split).

    Args:
        question: The evaluation question.
        chunk: One search result.

    Returns:
        True if the chunk holds the expected answer location.
    """
    metadata = chunk.metadata
    if metadata["source_path"] != question.source_path or metadata["section_title"] != question.section_title:
        return False
    if not question.subsection:
        return True
    return metadata["subsection"] == question.subsection or f"### {question.subsection}" in chunk.text


def rank_of_expected(question: EvalQuestion, results: list[RetrievedChunk]) -> int | None:
    """Find where the expected chunk landed in the results.

    Args:
        question: The evaluation question.
        results: Search results, best first.

    Returns:
        The 1-based rank of the first matching chunk, or None if it is missing.
    """
    for rank, chunk in enumerate(results, start=1):
        if is_expected_chunk(question, chunk):
            return rank
    return None
