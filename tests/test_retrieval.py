"""Retrieval: vector, BM25, RRF fusion, and hybrid search on the fixture docs."""

import pytest

import config.settings as settings
from retrieval.bm25_search import bm25_search
from retrieval.hybrid_search import hybrid_search
from retrieval.models import RetrievedChunk
from retrieval.rrf import fuse_rankings
from retrieval.vector_search import vector_search
from vectorstore.chroma_client import get_collection

# (question, expected section, expected subsection)
QUERIES = [
    ("How do I install the SDK?", "Installation", ""),
    ("Which transports do MCP servers support?", "Server Setup", "Transports"),
    ("How do remote servers authenticate clients?", "Server Setup", "Authentication"),
    ("How do I register a tool?", "Server Setup", "Tool Registration"),
    ("What does error -32601 mean?", "Troubleshooting", ""),
    ("get_or_create_collection", "Vector Store", ""),
    ("How many requests per minute can I make?", "Rate Limits", ""),
    ("What happens if I get HTTP 429?", "Rate Limits", ""),
    ("How do I deploy with Docker?", "Deployment", ""),
    ("MCP_AUTH_TOKEN", "Server Setup", "Authentication"),
]


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=chunk_id, metadata={}, score=0.0)


@pytest.mark.parametrize(("question", "section", "subsection"), QUERIES)
def test_hybrid_ranks_expected_section_first(ingested: None, question: str, section: str, subsection: str) -> None:
    top = hybrid_search(question)[0]
    assert (top.metadata["section_title"], top.metadata["subsection"]) == (section, subsection)


@pytest.mark.parametrize("question", ["get_or_create_collection", "MCP_AUTH_TOKEN", "error -32601"])
def test_bm25_finds_exact_identifiers(ingested: None, question: str) -> None:
    results = bm25_search(get_collection(), question, settings.TOP_K)
    assert results and any(part in results[0].text for part in question.split())


def test_vector_search_scores_are_similarities(ingested: None) -> None:
    results = vector_search(get_collection(), "installing the package", settings.TOP_K)
    assert len(results) == settings.TOP_K
    assert all(-1.0 <= c.score <= 1.0 for c in results)
    assert [c.score for c in results] == sorted((c.score for c in results), reverse=True)


def test_hybrid_returns_at_most_top_k_sorted(ingested: None) -> None:
    results = hybrid_search("installation")
    assert 0 < len(results) <= settings.TOP_K
    assert [c.score for c in results] == sorted((c.score for c in results), reverse=True)


def test_bm25_skips_chunks_with_no_shared_words(ingested: None) -> None:
    assert bm25_search(get_collection(), "zzzqqq xylophone", settings.TOP_K) == []


def test_rrf_rewards_chunks_found_by_both_searches() -> None:
    fused = fuse_rankings([[_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]])
    assert [c.chunk_id for c in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / (settings.RRF_K + 2) + 1 / (settings.RRF_K + 1))


def test_rrf_of_nothing_is_empty() -> None:
    assert fuse_rankings([[], []]) == []
