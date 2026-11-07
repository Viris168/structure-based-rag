"""Generation: prompt building, citations, and the LLM client's key check."""

import pytest

import config.settings as settings
import generation.llm_client as llm_client
from chunking.context_header import build_heading_path
from generation.citation import cited_sources, expand_citation_numbers, format_citation
from generation.prompt_builder import build_grounded_prompt
from retrieval.hybrid_search import hybrid_search
from retrieval.models import RetrievedChunk


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
    for number, chunk in enumerate(chunks, start=1):
        m = chunk.metadata
        heading_path = build_heading_path(str(m["doc_title"]), str(m["section_title"]), str(m["subsection"]))
        assert f"[{number}] {heading_path}" in user["content"]
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


# --- numbered citations --------------------------------------------------

def _numbered(*sections: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=section,
            text="text",
            metadata={"doc_title": "Docs", "section_title": section, "subsection": ""},
            score=0.0,
        )
        for section in sections
    ]


CHUNKS = _numbered("Installation", "Rate Limits", "Deployment")
INSTALL, RATES, DEPLOY = (format_citation(c.metadata) for c in CHUNKS)


def test_number_becomes_the_exact_label() -> None:
    assert expand_citation_numbers("Run pip install [1].", CHUNKS) == f"Run pip install {INSTALL}."


@pytest.mark.parametrize("markers", ["[1][3]", "[1, 3]", "[1] [3]", "[1][3][1]"])
def test_several_numbers_become_several_labels_once_each(markers: str) -> None:
    assert expand_citation_numbers(f"Both {markers}.", CHUNKS) == f"Both {INSTALL} {DEPLOY}."


def test_numbers_that_match_no_excerpt_are_dropped() -> None:
    assert expand_citation_numbers("Made up [7]. Mixed [2][9].", CHUNKS) == f"Made up . Mixed {RATES}."


def test_code_and_indexing_are_left_alone() -> None:
    answer = "Use `items[1]` [2].\n```python\nfirst = rows[1]\nprint([1])\n```\nThen call f(x)[1] or data[2]."
    expected = f"Use `items[1]` {RATES}.\n```python\nfirst = rows[1]\nprint([1])\n```\nThen call f(x)[1] or data[2]."
    assert expand_citation_numbers(answer, CHUNKS) == expected


def test_expanded_labels_are_found_as_sources() -> None:
    answer = expand_citation_numbers("Install [1]. Limits [2]. Install again [1].", CHUNKS)
    assert cited_sources(answer, CHUNKS) == [INSTALL, RATES]
