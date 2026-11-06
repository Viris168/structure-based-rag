"""Chunking: the H2/H3 rules, code fences, and the embedding token limit."""

from pathlib import Path

import config.settings as settings
from chunking.context_header import build_embedding_text, build_heading_path
from chunking.hybrid_chunker import chunk_document
from chunking.loader import load_markdown_files
from chunking.models import Chunk


def _chunks_by_doc(docs_dir: Path, split_on_h3: bool = True) -> dict[str, list[Chunk]]:
    return {d.source_path: chunk_document(d, split_on_h3) for d in load_markdown_files(docs_dir)}


def test_long_section_is_split_on_h3(docs_dir: Path) -> None:
    mcp = _chunks_by_doc(docs_dir)["MCP_Documentation.md"]
    subsections = [c.subsection for c in mcp if c.section_title == "Server Setup"]
    assert {"Transports", "Authentication", "Tool Registration"} <= set(subsections)


def test_section_intro_is_kept_when_h3_split(docs_dir: Path) -> None:
    mcp = _chunks_by_doc(docs_dir)["MCP_Documentation.md"]
    assert any(c.section_title == "Server Setup" and c.subsection == "" for c in mcp)


def test_short_sections_are_not_h3_split(docs_dir: Path) -> None:
    docs = _chunks_by_doc(docs_dir)["DOCUMENTATION.md"]
    assert all(c.subsection == "" for c in docs)


def test_h3_rule_can_be_switched_off(docs_dir: Path) -> None:
    mcp = _chunks_by_doc(docs_dir, split_on_h3=False)["MCP_Documentation.md"]
    assert all(c.subsection == "" for c in mcp)


def test_heading_inside_code_fence_is_ignored(docs_dir: Path) -> None:
    mcp = _chunks_by_doc(docs_dir)["MCP_Documentation.md"]
    assert not any("shell comment" in c.section_title for c in mcp)
    install = next(c for c in mcp if c.section_title == "Installation")
    assert "## this line is a shell comment" in install.text


def test_every_chunk_fits_the_embedding_model(docs_dir: Path) -> None:
    for chunks in _chunks_by_doc(docs_dir).values():
        for chunk in chunks:
            path = build_heading_path(chunk.doc_title, chunk.section_title, chunk.subsection)
            assert chunk.token_count <= settings.MAX_EMBED_TOKENS, path


def test_chunk_indexes_are_sequential(docs_dir: Path) -> None:
    for chunks in _chunks_by_doc(docs_dir).values():
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_preamble_uses_doc_title_as_section(docs_dir: Path) -> None:
    mcp = _chunks_by_doc(docs_dir)["MCP_Documentation.md"]
    assert mcp[0].section_title == "MCP Documentation"


def test_embedding_text_has_heading_path_unless_disabled(docs_dir: Path) -> None:
    chunk = next(c for c in _chunks_by_doc(docs_dir)["MCP_Documentation.md"] if c.subsection == "Transports")
    assert build_embedding_text(chunk).startswith("MCP Documentation > Server Setup > Transports\n\n")
    assert build_embedding_text(chunk, include_heading_path=False) == chunk.text
