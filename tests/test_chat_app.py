"""Chat UI: reply formatting, source links, and the document page (no browser or API key needed)."""

import gradio as gr
import pytest
from fastapi.testclient import TestClient

import config.settings as settings
from tests.conftest import FakeLLM
from ui.chat_app import build_chat_app, build_server, format_reply
from ui.doc_links import doc_url, heading_anchor
from ui.doc_page import render_doc_page

INSTALL_CITATION = "[Source: MCP Documentation · Section: Installation]"
AUTH_CITATION = "[Source: MCP Documentation · Section: Server Setup > Authentication]"


# --- chat replies --------------------------------------------------------

def test_reply_links_each_cited_source(ingested: None, fake_llm: FakeLLM) -> None:
    fake_llm.reply = f"Run pip install mcp-sdk. {INSTALL_CITATION}"
    reply = format_reply("How do I install the SDK?")
    assert reply.startswith("Run pip install mcp-sdk.")
    link = f"- [Source: MCP Documentation · Section: Installation]({settings.DOC_ROUTE}/MCP_Documentation.md#installation)"
    assert reply.endswith(f"**Sources**\n{link}")


def test_subsection_source_links_to_its_subsection(ingested: None, fake_llm: FakeLLM) -> None:
    fake_llm.reply = f"Use OAuth 2.1 bearer tokens. {AUTH_CITATION}"
    reply = format_reply("How do remote servers authenticate clients?")
    assert f"{settings.DOC_ROUTE}/MCP_Documentation.md#server-setup--authentication)" in reply


def test_refusal_has_no_sources_section(ingested: None, fake_llm: FakeLLM) -> None:
    assert format_reply("What is the capital of France?") == settings.REFUSAL_MESSAGE


def test_errors_are_shown_in_the_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.chat_app as chat_app

    def broken(question: str) -> None:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    monkeypatch.setattr(chat_app, "run_question", broken)
    assert format_reply("anything") == "**Error:** DEEPSEEK_API_KEY is not set."


def test_chat_app_builds() -> None:
    assert isinstance(build_chat_app(), gr.ChatInterface)


# --- links and document page ---------------------------------------------

def test_heading_anchor_includes_parent_section() -> None:
    assert heading_anchor("Server Setup", "Authentication") == "server-setup--authentication"
    assert heading_anchor("Uninstalling / Removal") == "uninstalling-removal"
    assert heading_anchor("Usage", "") == "usage"


def test_merged_chunk_links_to_first_subsection() -> None:
    metadata = {"source_path": "NVM README.md", "section_title": "Usage", "subsection": "Colors / Restoring PATH"}
    assert doc_url(metadata) == f"{settings.DOC_ROUTE}/NVM%20README.md#usage--colors"


def test_doc_page_has_the_anchors_links_point_to(docs_dir: object) -> None:
    page = render_doc_page("MCP_Documentation.md")
    assert page is not None
    for anchor in ("mcp-documentation", "installation", "server-setup", "server-setup--authentication"):
        assert f'id="{anchor}"' in page


def test_doc_page_shows_code_but_never_runs_html(docs_dir: object) -> None:
    page = render_doc_page("MCP_Documentation.md")
    assert page is not None and "<pre><code" in page
    assert 'id="this-line-is-a-shell-comment-not-a-heading"' not in page


@pytest.mark.parametrize("name", ["missing.md", "../.env", ".env", "../config/settings.py"])
def test_only_files_in_docs_can_be_shown(name: str) -> None:
    assert render_doc_page(name) is None


def test_server_serves_documents_and_404s_unknown_files(docs_dir: object) -> None:
    client = TestClient(build_server())
    response = client.get(f"{settings.DOC_ROUTE}/MCP_Documentation.md")
    assert response.status_code == 200 and 'id="installation"' in response.text
    assert client.get(f"{settings.DOC_ROUTE}/nope.md").status_code == 404
    assert client.get(f"{settings.DOC_ROUTE}/..%2F.env").status_code == 404
