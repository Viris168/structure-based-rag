"""Loads the evaluation questions and the section each one should retrieve."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuestion:
    """One test question and where its answer lives.

    Attributes:
        question: A plain-English question, phrased the way a user would ask it.
        source_path: Filename of the doc that answers it, e.g. "DOCUMENTATION.md".
        section_title: The `##` heading of the answering section.
        subsection: The `###` heading inside that section, or "" if the
            answer is anywhere in the section.
    """

    question: str
    source_path: str
    section_title: str
    subsection: str


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read the questions file.

    Each entry needs "question", "source_path", and "section_title";
    "subsection" is optional.

    Args:
        path: Path to the JSON file (a list of objects).

    Returns:
        The questions, in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no questions.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy eval/questions.example.json to eval/questions.json "
            "and write questions about your docs."
        )
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"{path} has no questions.")
    return [
        EvalQuestion(
            question=entry["question"],
            source_path=entry["source_path"],
            section_title=entry["section_title"],
            subsection=entry.get("subsection", ""),
        )
        for entry in entries
    ]
