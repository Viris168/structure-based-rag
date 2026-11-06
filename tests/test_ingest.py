"""Ingestion: first run, unchanged re-runs, edited docs, and deleted docs."""

import shutil
from pathlib import Path

import pytest

import vectorstore.chroma_client as chroma_client
from pipeline.ingest import run_ingestion
from retrieval.bm25_search import bm25_search
from retrieval.hybrid_search import hybrid_search
from vectorstore.chroma_client import get_collection


@pytest.fixture
def fresh_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point ingestion at a new, empty store for this test only."""
    monkeypatch.setattr(chroma_client, "CHROMA_PATH", tmp_path / "chroma")


def test_first_run_adds_every_doc(fresh_store: None) -> None:
    report = run_ingestion()
    assert sorted(report.added) == ["DOCUMENTATION.md", "MCP_Documentation.md"]
    assert report.chunks_written == get_collection().count() > 0


def test_unchanged_rerun_writes_nothing(fresh_store: None) -> None:
    run_ingestion()
    report = run_ingestion()
    assert len(report.skipped) == 2
    assert report.chunks_written == 0


def test_edited_doc_is_replaced(ingested: None, docs_dir: Path) -> None:
    path = docs_dir / "DOCUMENTATION.md"
    original = path.read_text()
    path.write_text(original.replace("60 requests per minute", "120 requests per minute"))
    try:
        report = run_ingestion()
        assert report.updated == ["DOCUMENTATION.md"]
        assert report.skipped == ["MCP_Documentation.md"]
        collection = get_collection()
        assert not collection.get(where_document={"$contains": "60 requests per minute"})["ids"]
        assert "120 requests" in bm25_search(collection, "120 requests", 1)[0].text
    finally:
        path.write_text(original)
        run_ingestion()


def test_deleted_doc_is_removed(ingested: None, docs_dir: Path, tmp_path: Path) -> None:
    path = docs_dir / "DOCUMENTATION.md"
    shutil.move(path, tmp_path / path.name)
    try:
        report = run_ingestion()
        assert report.removed == ["DOCUMENTATION.md"]
        results = hybrid_search("rate limits docker deployment")
        assert all(c.metadata["source_path"] != "DOCUMENTATION.md" for c in results)
    finally:
        shutil.move(tmp_path / path.name, path)
        run_ingestion()
