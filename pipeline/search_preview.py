"""Shows what hybrid search returns for a query, without calling the LLM."""

from generation.citation import format_citation
from retrieval.hybrid_search import hybrid_search


def preview_search(query: str) -> str:
    """Run hybrid search and describe each result.

    Args:
        query: A plain-English question about the docs.

    Returns:
        One line per result with its rank, reranker score, and citation label.
    """
    results = hybrid_search(query)
    if not results:
        return "No results. Run `python main.py ingest` first."
    return "\n".join(
        f"{rank}. ({chunk.score:.4f}) {format_citation(chunk.metadata)}"
        for rank, chunk in enumerate(results, start=1)
    )
