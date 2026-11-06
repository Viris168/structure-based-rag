"""Generation: prompt building, citations, and the LLM client's key check."""

import pytest

import config.settings as settings
import generation.llm_client as llm_client
from generation.citation import cited_sources, format_citation
from generation.prompt_builder import build_grounded_prompt
from retrieval.hybrid_search import hybrid_search


def test_citation_format() -> None:
    metadata = {"doc_title": "MCP Documentation", "section_title": "Server Setup", "subsection": "Transports"}
    assert format_citation(metadata) == "[Source: MCP Documentation · Section: Server Setup > Transports]"


def test_citation_without_subsection() -> None:
    metadata = {"doc_title": "Docs", "section_title": "Rate Limits", "subsection": ""}
    assert format_citation(metadata) == "[Source: Docs · Section: Rate Limits]"


def test_prompt_is_grounded_and_labelled(ingested: None) -> None:
    question = "How do I install the SDK?"
    chunks = hybrid_search(question)
    system, user = build_grounded_prompt(question, chunks)
    assert system["role"] == "system" and settings.REFUSAL_MESSAGE in system["content"]
    assert user["role"] == "user" and user["content"].endswith(f"Question: {question}")
    for chunk in chunks:
        assert format_citation(chunk.metadata) in user["content"]
    assert "pip install mcp-sdk" in user["content"]


def test_cited_sources_lists_each_cited_label_once(ingested: None) -> None:
    chunks = hybrid_search("rate limit")
    label = format_citation(chunks[0].metadata)
    assert cited_sources(f"Answer {label}. Again {label}.", chunks) == [label]


def test_cited_sources_ignores_answers_without_labels(ingested: None) -> None:
    assert cited_sources("No citations here.", hybrid_search("rate limit")) == []


def test_missing_api_key_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(settings.LLM_API_KEY_ENV, raising=False)
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: None)
    llm_client._client.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=settings.LLM_API_KEY_ENV):
            llm_client.generate_answer([{"role": "user", "content": "hi"}])
    finally:
        llm_client._client.cache_clear()
