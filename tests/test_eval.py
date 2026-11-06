"""Evaluation: answer matching, hit rate/MRR, the questions file, and the comparison."""

from pathlib import Path

import pytest

import config.settings as settings
import eval.hit_rate as hit_rate
from eval.compare import run_comparison
from eval.matching import is_expected_chunk, rank_of_expected
from eval.questions import EvalQuestion, load_questions
from retrieval.models import RetrievedChunk
from vectorstore.chroma_client import get_collection

QUESTION = EvalQuestion(question="q", source_path="A.md", section_title="Setup", subsection="Auth")


def _chunk(section: str, subsection: str, text: str = "", source: str = "A.md") -> RetrievedChunk:
    metadata = {"source_path": source, "section_title": section, "subsection": subsection}
    return RetrievedChunk(chunk_id=f"{section}/{subsection}", text=text, metadata=metadata, score=0.0)


# --- matching ------------------------------------------------------------

def test_exact_subsection_matches() -> None:
    assert is_expected_chunk(QUESTION, _chunk("Setup", "Auth"))


def test_unsplit_section_containing_subsection_matches() -> None:
    assert is_expected_chunk(QUESTION, _chunk("Setup", "", "## Setup\n### Auth\nUse tokens."))


def test_intro_chunk_of_split_section_does_not_match() -> None:
    assert not is_expected_chunk(QUESTION, _chunk("Setup", ""))


@pytest.mark.parametrize("chunk", [_chunk("Other", "Auth"), _chunk("Setup", "Auth", source="B.md")])
def test_wrong_section_or_doc_does_not_match(chunk: RetrievedChunk) -> None:
    assert not is_expected_chunk(QUESTION, chunk)


def test_section_only_question_matches_any_chunk_in_section() -> None:
    question = EvalQuestion(question="q", source_path="A.md", section_title="Setup", subsection="")
    assert is_expected_chunk(question, _chunk("Setup", "Anything"))


def test_rank_of_expected() -> None:
    results = [_chunk("Setup", "Other"), _chunk("Setup", "Auth")]
    assert rank_of_expected(QUESTION, results) == 2
    assert rank_of_expected(QUESTION, results[:1]) is None


# --- scoring -------------------------------------------------------------

def test_second_place_gives_mrr_half() -> None:
    result = hit_rate.score_search(lambda q: [_chunk("Setup", "Other"), _chunk("Setup", "Auth")], [QUESTION])
    assert result.hit_rate == 1.0 and result.mrr == 0.5


def test_miss_gives_zero() -> None:
    result = hit_rate.score_search(lambda q: [], [QUESTION])
    assert result.hit_rate == 0.0 and result.mrr == 0.0 and result.top_results == [None]


def test_run_eval_meets_target_on_fixtures(ingested: None) -> None:
    report = hit_rate.run_eval()
    assert "PASS" in report.splitlines()[1]


# --- questions file ------------------------------------------------------

def test_missing_questions_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="questions.example.json"):
        load_questions(tmp_path / "nope.json")


def test_empty_questions_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="no questions"):
        load_questions(path)


def test_example_questions_file_loads() -> None:
    questions = load_questions(settings.PROJECT_ROOT / "eval" / "questions.example.json")
    assert len(questions) == 3 and questions[0].subsection == ""


# --- comparison ----------------------------------------------------------

def test_comparison_scores_all_setups_without_touching_real_store(ingested: None) -> None:
    count_before = get_collection().count()
    lines = run_comparison().splitlines()
    rows = [line for line in lines if "%" in line]
    assert len(rows) == 7 and "(current)" in rows[-1]
    assert get_collection().count() == count_before
