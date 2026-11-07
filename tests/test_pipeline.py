"""The LangGraph question pipeline and the command line, with the LLM faked."""

import subprocess
import sys
from pathlib import Path

import pytest

import config.settings as settings
import vectorstore.chroma_client as chroma_client
from pipeline.ask import answer_question
from pipeline.search_preview import preview_search
from tests.conftest import FakeLLM

INSTALL_LABEL = "[Source: MCP Documentation · Section: Installation]"


def test_numbered_citation_becomes_label_and_source(ingested: None, fake_llm: FakeLLM) -> None:
    fake_llm.reply = "Run `pip install mcp-sdk` [1]."
    output = answer_question("How do I install the SDK?")
    answer, sources = output.split("Sources:")
    assert answer.strip() == f"Run `pip install mcp-sdk` {INSTALL_LABEL}."
    assert INSTALL_LABEL in sources


def test_llm_receives_grounded_prompt(ingested: None, fake_llm: FakeLLM) -> None:
    answer_question("How do I install the SDK?")
    assert len(fake_llm.calls) == 1
    assert "pip install mcp-sdk" in fake_llm.calls[0][1]["content"]


def test_model_refusal_is_passed_through(ingested: None, fake_llm: FakeLLM) -> None:
    fake_llm.reply = settings.REFUSAL_MESSAGE
    assert answer_question("What is the capital of France?") == settings.REFUSAL_MESSAGE


def test_empty_store_refuses_without_calling_llm(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(chroma_client, "CHROMA_PATH", tmp_path / "empty")
    assert answer_question("anything") == settings.REFUSAL_MESSAGE
    assert fake_llm.calls == []


def test_search_preview_shows_ranked_citations(ingested: None) -> None:
    first_line = preview_search("How do I deploy with Docker?").splitlines()[0]
    assert first_line.startswith("1. (") and "Section: Deployment]" in first_line


def test_search_preview_on_empty_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(chroma_client, "CHROMA_PATH", tmp_path / "empty")
    assert "Run `python main.py ingest` first" in preview_search("anything")


@pytest.mark.parametrize("command", ["ask", "search"])
def test_cli_requires_a_question(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "main.py", command],
        cwd=settings.PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "needs a question" in result.stderr
