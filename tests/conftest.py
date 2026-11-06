"""Shared test setup: point the project at fixture docs and a throwaway store.

Project modules copy settings at import time (`from config.settings import X`),
so the settings are patched here, at the top of conftest, before any test
module imports them. Nothing in tests/ ever touches docs/ or chroma_data/.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

import config.settings as settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_WORK_DIR = Path(tempfile.mkdtemp(prefix="rag_tests_"))

# Docs are copied so tests that edit or delete a doc never change the fixtures.
shutil.copytree(FIXTURES_DIR / "docs", _WORK_DIR / "docs")
settings.DOCS_DIR = _WORK_DIR / "docs"
settings.CHROMA_PATH = _WORK_DIR / "chroma"
settings.EVAL_QUESTIONS_PATH = FIXTURES_DIR / "questions.json"

from pipeline.ingest import run_ingestion  # noqa: E402  (must follow the patching above)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Delete the temporary docs and store after the run."""
    shutil.rmtree(_WORK_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    """The working copy of the fixture docs that tests ingest from."""
    return settings.DOCS_DIR


@pytest.fixture(scope="session")
def ingested() -> None:
    """Ingest the fixture docs once for every test that searches them."""
    run_ingestion()


class FakeLLM:
    """Stands in for the LLM: returns `reply` and records every prompt it receives."""

    def __init__(self) -> None:
        self.reply: str = settings.REFUSAL_MESSAGE
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Replace the LLM call in the graph with a FakeLLM (no API key needed)."""
    import pipeline.nodes as nodes

    fake = FakeLLM()
    monkeypatch.setattr(nodes, "generate_answer", fake)
    return fake
